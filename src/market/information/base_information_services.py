from typing import Dict
import numpy as np
from .information_types import InformationType, InformationSignal, InfoCapability, InformationProvider, SignalCategory, SIGNAL_CATEGORIES


class InformationService:
    """Central service managing all information distribution"""

    def __init__(self, agent_repository, market_state_managers=None):
        self.agent_repository = agent_repository
        self.market_state_managers = market_state_managers or {}
        # Multi-stock if market_state_managers dict was explicitly provided (even with 1 stock)
        # This must match base_sim.py's is_multi_stock = stock_configs is not None
        self.is_multi_stock = market_state_managers is not None and len(self.market_state_managers) > 0
        self.providers: Dict[InformationType, InformationProvider] = {}
        # Separate current signals from history
        self.current_signals: Dict[str, Dict[InformationType, InformationSignal]] = {}
        self.signal_history: Dict[int, Dict[str, Dict[InformationType, InformationSignal]]] = {}
        
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

            # Distribute to agents in multi-stock format
            for agent_id in self.agent_repository.get_all_agent_ids():
                agent = self.agent_repository.get_agent(agent_id)
                # For multi-stock, pass all stock signals
                agent_signals = {
                    'multi_stock_signals': all_stock_signals,
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
    
    def _modify_signal(self, signal: InformationSignal, capability: InfoCapability, round_number: int) -> InformationSignal:
        """Modify signal based on agent capabilities and signal category"""
        category = SIGNAL_CATEGORIES[signal.type]
        
        # 1. Handle PUBLIC signals (always pass through unchanged)
        if category == SignalCategory.PUBLIC:
            return signal
        
        # 2. Check if signal is enabled for this agent
        if not capability.enabled:
            return None
        
        # 3. Process by category
        value = signal.value
        reliability = signal.reliability * capability.accuracy
        metadata = signal.metadata.copy()
        
        if category == SignalCategory.MARKET:
            # Handle market data (e.g., order book depth)
            if capability.depth is not None and isinstance(value, dict):
                if 'buy_levels' in value:
                    value['buy_levels'] = value['buy_levels'][:capability.depth]
                if 'sell_levels' in value:
                    value['sell_levels'] = value['sell_levels'][:capability.depth]
                
        elif category == SignalCategory.FUNDAMENTAL:
            # Apply noise to fundamental signals
            if isinstance(value, (int, float)) and capability.noise_level > 0:
                noise = np.random.normal(0, capability.noise_level * abs(value))
                value += noise
                metadata['noisy'] = True
            
        elif category == SignalCategory.RESTRICTED:
            # Only pass if explicitly enabled with proper capability
            if not capability.enabled:
                return None
            
        # Apply common modifications
        if capability.delay > 0:
            metadata['original_round'] = round_number
            metadata['delay'] = capability.delay
        
        return InformationSignal(
            type=signal.type,
            value=value,
            reliability=reliability,
            duration=signal.duration,
            metadata=metadata
        )

    def _generate_agent_signals(self, agent, base_signals: Dict[InformationType, InformationSignal], 
                              round_number: int) -> Dict[InformationType, InformationSignal]:
        """Generate agent-specific signals based on their capabilities"""
        agent_signals = {}
        
        for info_type, base_signal in base_signals.items():
            if hasattr(agent, 'info_capabilities') and info_type in agent.info_capabilities:
                capability = agent.get_info_capability(info_type)
                if capability.enabled:
                    modified_signal = self._modify_signal(
                        base_signal, 
                        capability,
                        round_number
                    )
                    agent_signals[info_type] = modified_signal
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
