from agents.base_agent import BaseAgent
from agents.agent_types import AGENT_TYPES
import traceback
from typing import List, Dict, Any, Set
from .services.formatting_services import MarketStateFormatter, AgentContext
from services.logging_models import DecisionLogEntry
from .services.llm_services import LLMService, LLMRequest
from .services.schema_features import Feature, FeatureRegistry
from .services.prompt_builder import PromptBuilder
from services.logging_service import LoggingService
from market.information.information_types import InformationType

class LLMAgent(BaseAgent):
    # Memory system constants
    MEMORY_DISPLAY_LIMIT = 10  # Number of recent notes to show in prompts (prevents prompt bloat)

    def __init__(self, agent_id: str, agent_type: str,
                 model_open_ai: str = "gpt-oss-20b",
                 enabled_features: Set[Feature] = None,
                 *args, **kwargs):  # Usually set via scenario params
        super().__init__(agent_id, *args, **kwargs)
        self.agent_type = AGENT_TYPES[agent_type]
        self.model = model_open_ai
        self._formatter = MarketStateFormatter()
        self._llm_service = LLMService()

        # Feature toggle system: store enabled features
        # Default to all features for backward compatibility if not specified
        self.enabled_features = enabled_features if enabled_features is not None else FeatureRegistry.get_all_features()

        # Store last round's reasoning for continuity (always enabled)
        self.last_reasoning: Dict[str, Any] = {}  # {round, reasoning, valuation_reasoning, price_target_reasoning}

        # Conditionally initialize memory based on feature flags
        if Feature.MEMORY in self.enabled_features:
            self.memory_notes = []  # List of (round_number, note) tuples for agent memory

    def make_decision(self, market_state, history, round_number):
        try:
            # Store market_state for multi-stock support
            self.current_market_state = market_state

            # Prepare context using signals + market_state
            context = self.prepare_context_llm()

            # Build last reasoning section (if feature enabled - provides continuity)
            last_reasoning_section = ""
            if Feature.LAST_REASONING in self.enabled_features:
                last_reasoning_section = PromptBuilder.build_last_reasoning_section(
                    self.last_reasoning
                )

            # Build memory section using PromptBuilder (only if memory enabled)
            memory_section = ""
            if Feature.MEMORY in self.enabled_features:
                memory_notes = getattr(self, 'memory_notes', [])
                memory_section = PromptBuilder.build_memory_section(
                    memory_notes,
                    self.MEMORY_DISPLAY_LIMIT
                )

            # Build social feed section using PromptBuilder (only if social enabled)
            messages_section = ""
            if Feature.SOCIAL in self.enabled_features:
                last_messages = self.get_last_round_messages(round_number)
                messages_section = PromptBuilder.build_social_section(
                    last_messages,
                    self.enabled_features
                )

            # Detect if this is a multi-stock scenario
            is_multi_stock = 'stocks' in market_state

            # Build feature instructions (tells agent HOW to use memory/social)
            feature_instructions = PromptBuilder.build_instructions(self.enabled_features)

            user_prompt = (
                self.agent_type.user_prompt_template.format(**context)
                + last_reasoning_section
                + memory_section
                + messages_section
                + ("\n\n" + feature_instructions if feature_instructions else "")
            )

            # Create LLM request with enabled features
            request = LLMRequest(
                system_prompt=self.agent_type.system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                agent_id=self.agent_id,
                round_number=round_number,
                is_multi_stock=is_multi_stock,
                enabled_features=self.enabled_features
            )
            
            # Log prompt
            LoggingService.log_decision(
                f"\n========== Agent {self.agent_id} Prompt ==========\n"
                f"System: {request.system_prompt}\n"
                f"User: {request.user_prompt}"
            )
            
            # Get decision from LLM
            response = self._llm_service.get_decision(request)
            
            # Log raw response
            LoggingService.log_decision(
                f"\n========== Agent {self.agent_id} Response ==========\n"
                f"{response.raw_response}"
            )
            
            # Store replace_decision in the agent instance
            self.last_replace_decision = response.decision['replace_decision']

            # Store reasoning for next round's context
            self.last_reasoning = {
                'round': round_number,
                'reasoning': response.decision.get('reasoning', ''),
                'valuation_reasoning': response.decision.get('valuation_reasoning', ''),
                'price_target_reasoning': response.decision.get('price_target_reasoning', ''),
            }

            # Store memory notes with validation (only if memory feature enabled)
            if Feature.MEMORY in self.enabled_features:
                if note := response.decision.get('notes_to_self', '').strip():
                    # Ensure memory_notes exists (should be initialized in __init__)
                    if not hasattr(self, 'memory_notes'):
                        self.memory_notes = []

                    # Validation: warn if round numbers are non-monotonic
                    if self.memory_notes and round_number <= self.memory_notes[-1][0]:
                        LoggingService.log_decision(
                            f"\n========== Agent {self.agent_id} Memory Warning ==========\n"
                            f"Non-monotonic round numbers: previous={self.memory_notes[-1][0]}, current={round_number}"
                        )

                    # Store the note (all notes are kept in memory + saved to CSV)
                    self.memory_notes.append((round_number, note))

                    LoggingService.log_decision(
                        f"\n========== Agent {self.agent_id} Memory Note ==========\n"
                        f"Round {round_number}: {note}\n"
                        f"Total notes in memory: {len(self.memory_notes)}"
                    )

            # Get price signal for logging (handle multi-stock format)
            if isinstance(self.private_signals, dict) and self.private_signals.get('is_multi_stock'):
                # Multi-stock: get first stock's price signal
                first_stock_signals = next(iter(self.private_signals['multi_stock_signals'].values()))
                price_signal = first_stock_signals[InformationType.PRICE]
            else:
                # Single-stock: original behavior
                price_signal = self.private_signals[InformationType.PRICE]

            # Create and log structured decision entries
            log_entries = DecisionLogEntry.from_decision(
                decision=response.decision,
                agent_type_name=self.agent_type.name,
                agent_type_id=self.agent_type.type_id,
                round_number=round_number,
                market_price=price_signal.value
            )
            
            # Log each entry
            for entry in log_entries:
                LoggingService.log_structured_decision(entry)

            # Optional: Broadcast message if agent chose to post (only if social feature enabled)
            if Feature.SOCIAL in self.enabled_features:
                post_message = response.decision.get('post_message')
                message_reasoning = response.decision.get('message_reasoning')
                if post_message:
                    self.broadcast_message(round_number, post_message)
                    reasoning_text = f"\nMessage Reasoning: {message_reasoning}" if message_reasoning else ""
                    LoggingService.log_decision(
                        f"\n========== Agent {self.agent_id} Posted to Social Feed ==========\n"
                        f"{post_message}{reasoning_text}"
                    )

            # NOTE: stock_id auto-fix removed - now handled by dynamic schema
            # In single-stock mode, stock_id field is excluded from schema entirely
            # and automatically added in llm_services.py when parsing the response

            return response.decision
                
        except Exception as e:
            LoggingService.log_decision(
                f"\n========== Agent {self.agent_id} Error ==========\n"
                f"Error: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            fallback = self._llm_service.get_fallback_decision(
                agent_id=self.agent_id
            )
            LoggingService.log_decision(
                f"\n========== Agent {self.agent_id} Fallback ==========\n"
                f"Using fallback decision: {fallback}"
            )
            return fallback
    
    def prepare_agent_context(self):
        """Prepare agent context"""
        # Get all active orders (pending, active, and partially filled)
        active_orders = {
            'buy': (
                self.orders['pending']['buy'] + 
                self.orders['active']['buy'] + 
                self.orders['partially_filled']['buy']
            ),
            'sell': (
                self.orders['pending']['sell'] + 
                self.orders['active']['sell'] + 
                self.orders['partially_filled']['sell']
            )
        }
        
        # Check if multi-stock (more than just DEFAULT_STOCK)
        is_multi_stock = len(self.positions) > 1 or (
            len(self.positions) == 1 and "DEFAULT_STOCK" not in self.positions
        )

        # Calculate leverage metrics if leverage is enabled
        leverage_metrics = {}
        if self.leverage_ratio > 1.0 and hasattr(self, 'last_prices') and self.last_prices:
            try:
                leverage_metrics = {
                    'equity': self.get_equity(self.last_prices),
                    'gross_position_value': self.get_gross_position_value(self.last_prices),
                    'leverage_margin_ratio': self.get_leverage_margin_ratio(self.last_prices),
                    'available_borrowing_power': self.get_available_borrowing_power(self.last_prices),
                    'maintenance_margin': self.maintenance_margin
                }
            except:
                # If calculation fails, leave metrics as None
                pass

        return AgentContext(
            agent_id=self.agent_id,
            cash=self.cash,
            shares=self.shares,
            available_cash=self.available_cash,
            available_shares=self.available_shares,
            outstanding_orders={
                'buy': [
                    {
                        'quantity': order.remaining_quantity,
                        'price': order.price
                    }
                    for order in active_orders['buy']
                ],
                'sell': [
                    {
                        'quantity': order.remaining_quantity,
                        'price': order.price
                    }
                    for order in active_orders['sell']
                ]
            },
            signal_history=self.signal_history,
            trade_history=self.trade_history,
            dividend_cash=self.dividend_cash,
            committed_cash=self.committed_cash,
            committed_shares=self.committed_shares,
            # Multi-stock support
            positions=self.positions.copy() if is_multi_stock else None,
            available_positions={
                stock_id: self.positions[stock_id] for stock_id in self.positions
            } if is_multi_stock else None,
            committed_positions=self.committed_positions.copy() if is_multi_stock else None,
            is_multi_stock=is_multi_stock,
            # Leverage support
            borrowed_cash=self.borrowed_cash,
            leverage_ratio=self.leverage_ratio,
            leverage_interest_paid=self.leverage_interest_paid,
            # Short selling support
            allow_short_selling=self.allow_short_selling,
            **leverage_metrics  # Include calculated metrics if available
        )

    def prepare_context_llm(self):
        """Prepare context for LLM prompt using signals"""
        # Get market_state if available (for multi-stock support)
        market_state = getattr(self, 'current_market_state', None)

        return self._formatter.format_prompt_sections(
            agent_signals=self.private_signals,
            agent_context=self.prepare_agent_context(),
            signal_history=self.signal_history,
            market_state=market_state
        )

    def reset_memory(self):
        """Clear agent's memory notes (for simulation resets or testing)

        Note: This only clears the in-memory list. Notes are still preserved
        in CSV logs for permanent storage and analysis.
        Only works if memory feature is enabled.
        """
        if Feature.MEMORY in self.enabled_features and hasattr(self, 'memory_notes'):
            self.memory_notes.clear()
            LoggingService.log_decision(
                f"\n========== Agent {self.agent_id} Memory Reset ==========\n"
                f"All memory notes cleared"
            )
        else:
            LoggingService.log_decision(
                f"\n========== Agent {self.agent_id} Memory Reset Skipped ==========\n"
                f"Memory feature not enabled for this agent"
            )

    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """Export memory notes as structured data for analysis

        Returns:
            List of dicts with keys: round, note
            Empty list if memory feature is not enabled
        """
        if Feature.MEMORY in self.enabled_features and hasattr(self, 'memory_notes'):
            return [
                {'round': round_num, 'note': note}
                for round_num, note in self.memory_notes
            ]
        return []
 