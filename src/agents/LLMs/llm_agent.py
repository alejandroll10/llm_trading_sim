from agents.base_agent import BaseAgent
from agents.agent_types import AGENT_TYPES
import traceback
from .services.formatting_services import MarketStateFormatter, AgentContext
from services.logging_models import DecisionLogEntry
from .services.llm_services import LLMService, LLMRequest
from services.logging_service import LoggingService
from market.information.information_types import InformationType

class LLMAgent(BaseAgent):
    def __init__(self, agent_id: str, agent_type: str,
                 model_open_ai: str = "gpt-oss-20b", *args, **kwargs):  # Usually set via scenario params
        super().__init__(agent_id, *args, **kwargs)
        self.agent_type = AGENT_TYPES[agent_type]
        self.model = model_open_ai
        self._formatter = MarketStateFormatter()
        self._llm_service = LLMService()
        self.memory_notes = []  # List of (round_number, note) tuples for agent memory

    def make_decision(self, market_state, history, round_number):
        try:
            # Store market_state for multi-stock support
            self.current_market_state = market_state

            # Prepare context using signals + market_state
            context = self.prepare_context_llm()

            # Get last round messages for social feed context
            last_messages = self.get_last_round_messages(round_number)

            strategic_instructions = """
MEMORY SYSTEM:
You can optionally write notes to yourself in the 'notes_to_self' field - these will be shown to you in future rounds.
Use this to track patterns, record what works/doesn't work, and improve your strategy over time.

MESSAGING:
You can optionally post a message visible to all agents next round using the 'post_message' field.
Before posting, explain your intent in 'message_reasoning' - what effect do you want your message to have?

Strategic Considerations:
- Messages can influence other agents' beliefs and decisions
- You may share information to shape market sentiment
- You may withhold information for competitive advantage
- You may signal confidence, uncertainty, or specific views to move prices
- Consider: What do you want other agents to believe?
- Be explicit about your messaging strategy in 'message_reasoning'"""

            if last_messages:
                formatted_messages = "\n".join(
                    f"- Agent {m['agent_id']}: {m['message']}"
                    for m in last_messages
                )
                messages_section = f"\n\nSocial Feed (previous round):\n{formatted_messages}\n{strategic_instructions}"
            else:
                messages_section = f"\n\nSocial Feed: No messages yet.\n{strategic_instructions}"

            # Build memory section from stored notes
            if self.memory_notes:
                recent_notes = self.memory_notes[-10:]  # Last 10 notes
                memory_section = "\n\n=== YOUR MEMORY LOG (Last Notes) ===\n" + "\n".join(
                    f"Round {r}: {note}" for r, note in recent_notes
                ) + "\n"
            else:
                memory_section = ""

            # Detect if this is a multi-stock scenario
            is_multi_stock = 'stocks' in market_state

            user_prompt = (
                self.agent_type.user_prompt_template.format(**context)
                + memory_section
                + messages_section
            )

            # Create LLM request
            request = LLMRequest(
                system_prompt=self.agent_type.system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                agent_id=self.agent_id,
                round_number=round_number,
                is_multi_stock=is_multi_stock
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

            # Store memory notes
            if note := response.decision.get('notes_to_self', '').strip():
                self.memory_notes.append((round_number, note))
                LoggingService.log_decision(
                    f"\n========== Agent {self.agent_id} Memory Note ==========\n"
                    f"Round {round_number}: {note}"
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

            # Optional: Broadcast message if agent chose to post
            post_message = response.decision.get('post_message')
            message_reasoning = response.decision.get('message_reasoning')
            if post_message:
                self.broadcast_message(round_number, post_message)
                reasoning_text = f"\nMessage Reasoning: {message_reasoning}" if message_reasoning else ""
                LoggingService.log_decision(
                    f"\n========== Agent {self.agent_id} Posted to Social Feed ==========\n"
                    f"{post_message}{reasoning_text}"
                )

            # Auto-fix stock_id for single-stock scenarios
            # In single-stock scenarios, market_state won't have 'stocks' key
            is_single_stock = 'stocks' not in market_state
            if is_single_stock and response.decision.get('orders'):
                for order in response.decision['orders']:
                    if isinstance(order, dict):
                        order['stock_id'] = 'DEFAULT_STOCK'

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
 