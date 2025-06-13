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
                 model_open_ai: str = "gpt-4o", *args, **kwargs):
        super().__init__(agent_id, *args, **kwargs)
        self.agent_type = AGENT_TYPES[agent_type]
        self.model = model_open_ai
        self._formatter = MarketStateFormatter()
        self._llm_service = LLMService()

    def make_decision(self, market_state, history, round_number):
        try:
            # Use signals instead of market_state
            context = self.prepare_context_llm()
            
            # Create LLM request
            request = LLMRequest(
                system_prompt=self.agent_type.system_prompt,
                user_prompt=self.agent_type.user_prompt_template.format(**context),
                model=self.model,
                agent_id=self.agent_id,
                round_number=round_number
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
            
            # Get price signal for logging
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
            committed_shares=self.committed_shares
        )

    def prepare_context_llm(self):
        """Prepare context for LLM prompt using signals"""
        return self._formatter.format_prompt_sections(
            agent_signals=self.private_signals,
            agent_context=self.prepare_agent_context(),
            signal_history=self.signal_history
        )
 