from services.logging_models import LogFormatter
from agents.agent_manager.agent_repository import AgentRepository
from agents.agent_manager.services.commitment_services import CommitmentCalculator
from agents.agent_manager.services.position_services import PositionCalculator
from market.orders.order import Order, OrderState
from services.logging_service import LoggingService
from agents.agent_manager.agent_repository import AgentStateSnapshot

class AgentManager:
    """Base manager with core functionality and shared state"""
    def __init__(self, agent_repository: AgentRepository, context, market_state_manager, 
                 decisions_logger, agents_logger,
                 order_repository, order_state_manager, order_book=None,
                 position_calculator=None,
                 commitment_calculator=None,
                 log_formatter=None):
        self.context = context
        self.market_state_manager = market_state_manager
        self.decisions_logger = decisions_logger
        self.agents_logger = agents_logger
        self.order_repository = order_repository
        self.order_state_manager = order_state_manager
        self.order_book = order_book
        
        # Store repository directly
        self.agent_repository = agent_repository
        
        # Use injected services or create new ones
        self._position_calculator = position_calculator or PositionCalculator()
        self._commitment_calculator = commitment_calculator or CommitmentCalculator(order_book)
        self._log_formatter = log_formatter or LogFormatter()

    def validate_order(self, order: Order) -> tuple[bool, float, str]:
        """Validate order resources without changing state"""
        # Check if agent exists
        if not self.agent_repository.get_commitment_state(order.agent_id):
            return False, 0, f"Agent {order.agent_id} not found"

        # Validate using commitment manager
        valid, message = self.validate_commitment(
            order, 
            self.context.current_price
        )
        
        return valid, order.quantity if valid else 0, message

    def verify_agent_states(self):
        """Verify all agent states are consistent"""
        self.agents_logger.info("\n=== Verifying Agent States ===")
        
        # Use repository to verify all states
        try:
            if not self.agent_repository.verify_all_states():
                raise ValueError("Agent state verification failed")
            self.agents_logger.info("✓ All agent states verified")
        except ValueError as e:
            self.agents_logger.error(f"❌ State verification failed: {str(e)}")
            raise

    def verify_single_agent(self, agent_id: str, pre_state: AgentStateSnapshot):
        """Verify single agent state consistency"""
        current_state = self.agent_repository.get_agent_state_snapshot(
            agent_id,
            self.context.current_price
        )
        
        self.agents_logger.info(f"\nVerifying Agent {agent_id} State:")
        self._log_state_comparison(pre_state, current_state)
        
        try:
            self._verify_state_invariants(current_state)
            self.agents_logger.info("✓ State verification passed")
        except AssertionError as e:
            self.agents_logger.error(f"❌ State verification failed: {str(e)}")
            raise

    def _verify_state_invariants(self, state: AgentStateSnapshot):
        """Verify state invariants"""
        assert state.committed_cash >= 0, "Negative committed cash"
        assert state.committed_shares >= 0, "Negative committed shares"
        assert state.cash >= 0, "Negative cash"
        assert state.shares >= 0, "Negative shares"
        assert state.committed_cash <= state.cash + state.committed_cash, "Excess committed cash"
        assert state.committed_shares <= state.shares + state.committed_shares, "Excess committed shares"

    def _log_state_comparison(self, pre: AgentStateSnapshot, current: AgentStateSnapshot):
        """Log comparison between pre and current states"""
        self.agents_logger.info("Pre-round state:")
        self._log_state(pre, "Pre-round ")
        
        self.agents_logger.info("Current state:")
        self._log_state(current, "Current ")

    def _log_state(self, state: AgentStateSnapshot, prefix: str = ""):
        """Log single state snapshot"""
        self.agents_logger.info(f"{prefix}Cash: ${state.cash:.2f}")
        self.agents_logger.info(f"{prefix}Shares: {state.shares}")
        self.agents_logger.info(f"{prefix}Committed Cash: ${state.committed_cash:.2f}")
        self.agents_logger.info(f"{prefix}Committed Shares: {state.committed_shares}")

    def validate_commitment(self, order: Order, current_price: float) -> tuple[bool, str]:
        """Validate if agent has sufficient resources for commitment"""
        required = self._commitment_calculator.calculate_required_commitment(order, current_price)
        
        # Consider existing commitment if partially filled
        if order.state in [OrderState.PARTIALLY_FILLED, OrderState.ACTIVE]:
            required -= order.remaining_commitment
            
        # Get commitment state through repository
        state = self.agent_repository.get_commitment_state(order.agent_id)
        agent = self.agent_repository.get_agent(order.agent_id)
        
        if order.side == 'buy':
            if required > state.available_cash + order.current_cash_commitment:
                error_msg = f"Insufficient cash: needs {required:.2f}, has {state.available_cash:.2f} + {order.current_cash_commitment:.2f}"
                LoggingService.log_validation_error(
                    round_number=self.agent_repository.current_round,
                    agent_id=order.agent_id,
                    agent_type=agent.__class__.__name__,
                    error_type="INSUFFICIENT_CASH",
                    details=error_msg,
                    attempted_action=f"BUY {order.quantity} @ {current_price:.2f}"
                )
                return False, error_msg
        else:
            if required > state.available_shares + order.current_share_commitment:
                error_msg = f"Insufficient shares: needs {required}, has {state.available_shares} available shares + {order.current_share_commitment} currently committed shares"
                # Log validation error
                LoggingService.log_validation_error(
                    round_number=self.agent_repository.current_round,
                    agent_id=order.agent_id,
                    agent_type=agent.__class__.__name__,
                    error_type="INSUFFICIENT_SHARES",
                    details=error_msg,
                    attempted_action=f"SELL {order.quantity} @ {current_price:.2f}"
                )
                # Log agent state
                LoggingService.log_agent_state(
                    agent_id=order.agent_id,
                    operation=f"Verification commitment failed. {error_msg}",
                    agent_state=agent._get_state_dict(),
                    outstanding_orders=agent.outstanding_orders,
                    order_history=agent.order_history,
                    print_to_terminal=True,
                    is_error=True
                )
                return False, error_msg
                
        return True, "Valid"
