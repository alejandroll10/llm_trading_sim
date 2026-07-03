from typing import Dict, Optional
import numpy as np
from .information_types import InformationType, InformationSignal, InfoCapability, InformationProvider, SignalCategory, SIGNAL_CATEGORIES
from .info_capability_config import build_own_quality_text, summarize_others_quality


class InformationService:
    """Central service managing all information distribution"""

    def __init__(self, agent_repository, market_state_managers=None, info_capabilities_config=None):
        self.agent_repository = agent_repository
        self.market_state_managers = market_state_managers or {}
        # Multi-stock if market_state_managers dict was explicitly provided (even with 1 stock)
        # This must match base_sim.py's is_multi_stock = stock_configs is not None
        self.is_multi_stock = market_state_managers is not None and len(self.market_state_managers) > 0
        self.providers: Dict[InformationType, InformationProvider] = {}
        # Separate current signals from history
        self.current_signals: Dict[str, Dict[InformationType, InformationSignal]] = {}
        self.signal_history: Dict[int, Dict[str, Dict[InformationType, InformationSignal]]] = {}
        # Scenario-level per-agent signal config (drives prompt disclosure toggles).
        self.info_capabilities_config = info_capabilities_config or {}
        self.disclose_signal_quality = bool(self.info_capabilities_config.get('disclose_signal_quality', False))
        self.disclose_others_quality = bool(self.info_capabilities_config.get('disclose_others_quality', False))
        # Lazily-computed common-knowledge summary of every agent's signal noise.
        self._others_quality_summary: Optional[str] = None
        
    def register_provider(self, type: InformationType, provider: InformationProvider):
        """Register an information provider"""
        self.providers[type] = provider
    
    def distribute_information(self, round_number: int):
        """Generate and distribute all information signals"""
        if self.is_multi_stock:
            # Multi-stock: Generate signals for each stock
            all_stock_signals = {}
            for stock_id, manager in self.market_state_managers.items():
                stock_signals = {
                    info_type: provider.generate_signal_for_manager(manager, round_number)
                    for info_type, provider in self.providers.items()
                }
                all_stock_signals[stock_id] = stock_signals

            # Store in history
            self.signal_history[round_number] = {
                'base': all_stock_signals,  # Store all stocks
                'agent': {}
            }

            # Clear current signals
            self.current_signals.clear()

            # Distribute to agents in multi-stock format, applying each agent's
            # per-signal capabilities (noise/delay/depth/accuracy) per stock.
            for agent_id in self.agent_repository.get_all_agent_ids():
                agent = self.agent_repository.get_agent(agent_id)
                if getattr(agent, 'info_capabilities', None):
                    # Build a private, capability-modified copy for each stock so
                    # asymmetric-information designs also work in multi-stock mode.
                    agent_stock_signals = {
                        stock_id: self._generate_agent_signals(agent, stock_signals, round_number, stock_id)
                        for stock_id, stock_signals in all_stock_signals.items()
                    }
                else:
                    # No capabilities configured: share the base signals unchanged.
                    agent_stock_signals = all_stock_signals
                agent_signals = {
                    'multi_stock_signals': agent_stock_signals,
                    'is_multi_stock': True
                }
                self.current_signals[agent_id] = agent_signals
                self.signal_history[round_number]['agent'][agent_id] = agent_signals

            # Distribute to agents
            self.agent_repository.distribute_information(self.current_signals)
        else:
            # Single stock: Original behavior
            # Generate base signals
            base_signals = {
                info_type: provider.generate_signal(round_number)
                for info_type, provider in self.providers.items()
            }

            # Store in history
            self.signal_history[round_number] = {
                'base': base_signals,
                'agent': {}
            }

            # Clear current signals
            self.current_signals.clear()

            # Distribute to agents
            for agent_id in self.agent_repository.get_all_agent_ids():
                agent = self.agent_repository.get_agent(agent_id)
                agent_signals = self._generate_agent_signals(
                    agent, base_signals, round_number
                )
                self.current_signals[agent_id] = agent_signals
                self.signal_history[round_number]['agent'][agent_id] = agent_signals

            # Distribute to agents
            self.agent_repository.distribute_information(self.current_signals)

        
    def get_private_info(self, agent_id: str = None):
        """Get agent-specific information"""
        if agent_id is None:
            return self.current_signals
        
        if agent_id not in self.current_signals:
            raise ValueError(f"No signals for agent: {agent_id}")
        
        return self.current_signals[agent_id]
    
    def _get_delayed_base_signal(self, info_type: InformationType, round_number: int,
                                 delay: int, stock_id: Optional[str]) -> Optional[InformationSignal]:
        """Return the base signal for ``info_type`` from ``round_number - delay``.

        Looks the signal up in ``signal_history`` (which is populated for the
        current round BEFORE agent signals are generated, so past rounds are
        always present once they exist). Returns ``None`` during the first
        ``delay`` rounds — before any history for the target round exists — so
        callers fall back to serving the current (freshest) signal.

        Handles both layouts: single-stock ``history['base'][info_type]`` and
        multi-stock ``history['base'][stock_id][info_type]``.
        """
        target_round = round_number - delay
        if target_round < 0:
            return None
        history = self.signal_history.get(target_round)
        if not history:
            return None
        base = history.get('base')
        if base is None:
            return None
        if self.is_multi_stock:
            stock_base = base.get(stock_id)
            if not stock_base:
                return None
            return stock_base.get(info_type)
        return base.get(info_type)

    def _modify_signal(self, signal: InformationSignal, capability: InfoCapability,
                       round_number: int, stock_id: Optional[str] = None) -> InformationSignal:
        """Modify signal based on agent capabilities and signal category.

        Capability application order (documented, order matters):
          1. DELAY (first): if ``capability.delay > 0`` and a signal from
             ``round_number - delay`` exists, serve that stale round's VALUE.
             Only the value is staled; the CURRENT round's structural metadata
             (``round``, ``periods_remaining``, ``redemption_value``) is kept,
             because downstream horizon math combines ``periods_remaining`` with
             the undelayed public price round (``formatting_services``), and a
             stale ``periods_remaining`` would corrupt that horizon. The agent
             thus gets a delayed *estimate of value* while still knowing the true
             time-to-redemption.
          2. NOISE / ACCURACY (second): the agent's own ``noise_level`` and
             ``accuracy`` are then realized NOW on the (possibly stale) value —
             modelling a latency-limited instrument reading old data, rather than
             re-realizing the noise at the original round.
        """
        category = SIGNAL_CATEGORIES[signal.type]

        # 1. Handle PUBLIC signals (always pass through unchanged)
        if category == SignalCategory.PUBLIC:
            return signal

        # 2. Check if signal is enabled for this agent
        if not capability.enabled:
            return None

        # 3. Apply DELAY first: serve a stale VALUE from `round - delay` if that
        # round's history exists. Structural metadata stays current (see docstring).
        delayed_from_round = None
        value = signal.value
        if capability.delay > 0:
            delayed = self._get_delayed_base_signal(
                signal.type, round_number, capability.delay, stock_id
            )
            if delayed is not None:
                value = delayed.value
                delayed_from_round = round_number - capability.delay

        # 4. Process by category
        reliability = signal.reliability * capability.accuracy
        metadata = signal.metadata.copy()
        # Record the applied capability so it can be logged and (optionally)
        # disclosed to the agent, even when noise_level is 0.
        metadata['noise_level'] = capability.noise_level
        metadata['accuracy'] = capability.accuracy

        if category == SignalCategory.MARKET:
            # Handle market data (e.g., order book depth). Always copy the dict
            # so the served value never aliases the shared base signal — the same
            # value object is handed to every agent, and under `delay > 0` it is
            # the dict stored in a past round's `signal_history['base']` entry
            # (see #99). The copy runs regardless of `depth` so a config combining
            # `order_book` + `delay` cannot leak a mutable reference into history.
            if isinstance(value, dict):
                value = dict(value)
                # Rebuild the level lists as new list objects too, so list-level
                # mutation (append/replace) can't reach back into history either.
                # `depth is None` copies the full list; otherwise it truncates —
                # slicing produces a fresh list in both cases.
                depth = capability.depth
                for side in ('buy_levels', 'sell_levels'):
                    if side in value:
                        value[side] = value[side][:depth] if depth is not None else list(value[side])

        elif category == SignalCategory.FUNDAMENTAL:
            # Apply noise to fundamental signals
            if isinstance(value, (int, float)) and capability.noise_level > 0:
                noise = np.random.normal(0, capability.noise_level * abs(value))
                metadata['true_value'] = value  # keep the un-noised value for scoring
                value += noise
                metadata['noisy'] = True

        elif category == SignalCategory.RESTRICTED:
            # Only pass if explicitly enabled with proper capability
            if not capability.enabled:
                return None

        # Apply common modifications
        if capability.delay > 0:
            metadata['delay'] = capability.delay
            metadata['current_round'] = round_number
            if delayed_from_round is not None:
                # Served a stale value from an earlier round.
                metadata['original_round'] = delayed_from_round
                metadata['is_stale'] = True
            else:
                # First `delay` rounds: no history yet, so the current (fresh)
                # value was served. Record this so logs don't misread it as stale.
                metadata['original_round'] = round_number
                metadata['is_stale'] = False

        # Optional prompt disclosure of this agent's own signal quality.
        if self.disclose_signal_quality:
            metadata['quality_disclosure'] = build_own_quality_text(signal.type, capability)
        if self.disclose_others_quality:
            summary = self._get_others_quality_summary()
            if summary:
                metadata['others_quality_disclosure'] = summary

        return InformationSignal(
            type=signal.type,
            value=value,
            reliability=reliability,
            duration=signal.duration,
            metadata=metadata
        )

    def _get_others_quality_summary(self) -> str:
        """Build (once) the common-knowledge summary of every agent's signal noise."""
        if self._others_quality_summary is not None:
            return self._others_quality_summary
        noise_levels = []
        agent_ids = self.agent_repository.get_all_agent_ids()
        for agent_id in agent_ids:
            other = self.agent_repository.get_agent(agent_id)
            cap = None
            if getattr(other, 'info_capabilities', None):
                cap = other.info_capabilities.get(InformationType.FUNDAMENTAL)
            noise_levels.append(cap.noise_level if cap else 0.0)
        self._others_quality_summary = summarize_others_quality(noise_levels, len(agent_ids))
        return self._others_quality_summary

    def _generate_agent_signals(self, agent, base_signals: Dict[InformationType, InformationSignal],
                              round_number: int, stock_id: Optional[str] = None) -> Dict[InformationType, InformationSignal]:
        """Generate agent-specific signals based on their capabilities.

        ``stock_id`` identifies which stock these base signals belong to (None in
        single-stock mode); it is threaded to ``_modify_signal`` so the delay
        capability can look the stale value up in the correct per-stock history.
        """
        agent_signals = {}

        for info_type, base_signal in base_signals.items():
            if hasattr(agent, 'info_capabilities') and info_type in agent.info_capabilities:
                capability = agent.get_info_capability(info_type)
                modified_signal = self._modify_signal(
                    base_signal,
                    capability,
                    round_number,
                    stock_id
                )
                if modified_signal is not None:
                    agent_signals[info_type] = modified_signal
                else:
                    # Signal disabled for this agent. Keep the KEY (never drop it,
                    # or downstream formatters that index signals directly raise
                    # KeyError and the agent silently falls back to "hold" forever).
                    # Hide the value but preserve metadata so structural fields
                    # (periods_remaining, round, ...) survive and formatting shows
                    # the value as "Unavailable".
                    agent_signals[info_type] = InformationSignal(
                        type=base_signal.type,
                        value=None,
                        reliability=0.0,
                        duration=base_signal.duration,
                        metadata=base_signal.metadata.copy()
                    )
            else:
                # If agent has no specific capability, pass signal unchanged
                agent_signals[info_type] = base_signal

        return agent_signals

    def get_signal_history(self, round_number: int = None, agent_id: str = None):
        """Get historical signals"""
        if round_number is None:
            return self.signal_history
            
        if round_number not in self.signal_history:
            raise ValueError(f"No signals for round: {round_number}")
            
        if agent_id:
            return self.signal_history[round_number]['agent'].get(agent_id)
        return self.signal_history[round_number]['base']
