from typing import Tuple, Optional, List, Dict
from market.orders.order import Order, OrderState
from market.orders.order_repository import OrderRepository
import logging
from dataclasses import dataclass, field
from agents.agent_manager.services.commitment_services import CommitmentCalculator, release_for_cancellation

@dataclass
class StateTransitionResult:
    success: bool
    message: str
    order: Order
    from_state: OrderState
    to_state: OrderState

@dataclass
class StateTransitionMetrics:
    total_transitions: int = 0
    failed_transitions: int = 0
    transition_times: Dict[Tuple[OrderState, OrderState], float] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)

class OrderStateManager:
    """Centralized manager for order state transitions and associated side effects.
    """
    
    def __init__(self, 
                 order_repository: OrderRepository,
                 agent_repository,
                 order_book,
                 logger: Optional[logging.Logger],
                 commitment_calculator: CommitmentCalculator):
        self.order_repository = order_repository
        self.agent_repository = agent_repository
        self.order_book = order_book
        self.logger = logger or logging.getLogger('order_state')
        self.commitment_calculator = commitment_calculator
        self.metrics = StateTransitionMetrics()

    def handle_new_order(self, order: Order, current_price: float) -> Tuple[bool, str]:
        """Handle new order validation, commitment, and initial state transition"""
        try:
            # Calculate required commitment
            required_cash = order.quantity * current_price if order.side == 'buy' else 0
            required_shares = order.quantity if order.side == 'sell' else 0
            
            # Commit resources through repository
            if order.side == 'buy':
                result = self.agent_repository.commit_resources(
                    order.agent_id, 
                    cash_amount=required_cash
                )
            else:
                result = self.agent_repository.commit_resources(
                    order.agent_id, 
                    share_amount=required_shares
                )

            if result.success:
                # Transition to VALIDATED state
                self.order_repository.transition_state(
                    order.order_id, 
                    OrderState.VALIDATED,
                    notes=f"Passed validations"
                )
                # Update order commitments
                if order.side == 'buy':
                    order.original_cash_commitment = required_cash
                    order.current_cash_commitment = required_cash
                else:
                    order.original_share_commitment = required_shares
                    order.current_share_commitment = required_shares
                
                # Transition to COMMITTED state
                self.order_repository.transition_state(
                    order.order_id, 
                    OrderState.COMMITTED,
                    notes=f"Committed: {result.committed_amount:.2f}"
                )                    

                self.logger.info(f"Order {order.order_id} validated and committed")
                self.logger.info(f"Order {order.order_id} has history: {order.print_history()}")
                self.sync_agent_orders(order.agent_id)
                
                return True, "Order validated and committed successfully"
            else:
                # Failed validation - transition to CANCELLED
                self.order_repository.transition_state(
                    order.order_id, 
                    OrderState.CANCELLED,
                    notes=f"Validation failed: {result.message}"
                )

                self.logger.warning(f"Order {order.order_id} with history {order.print_history()} \n failed validation: {result.message}")
                return False, result.message
                
        except Exception as e:
            self.logger.error(f"Order validation failed: {str(e)}")
            return False, str(e)


    def handle_agent_all_orders_cancellation(self, agent_id: str, orders: List[Order],
                             message: str = "Cancelled"):
        """Single entry point for cancellation including commitment release and state transition and order book removal"""
        for order in orders:
            self.handle_single_order_cancellation(order, message, skip_sync=True)
        
        # Single sync at the end for efficiency
        self.sync_agent_orders(agent_id)

    def handle_single_order_cancellation(self, order: Order, message: str = "Cancelled", skip_sync: bool = False):
        """Cancel a single order"""

        # Transition to CANCELLED state
        self.order_repository.transition_state(
            order.order_id, 
            OrderState.CANCELLED,
            notes=f"Cancelled: {message}"
        )
    
        # Release commitments
        release_for_cancellation(
            agent_repository=self.agent_repository, 
            logger=self.logger, 
            order=order
        )

        # Remove from book
        self.order_book._remove_order_from_book(order)
        
        if not skip_sync:
            self.sync_agent_orders(order.agent_id)

    def cancel_existing_orders(self, new_orders: List[Order], order_book, message: str = "Cancelled"):
        """Cancel all existing orders for agents submitting new orders"""
        agents_with_new_orders = set(order.agent_id for order in new_orders)
        
        for agent_id in agents_with_new_orders:
            orders_in_book = self.order_repository.get_book_orders_from_agent(agent_id)
            
            # Use the centralized cancellation handler
            self.handle_agent_all_orders_cancellation(
                agent_id=agent_id,
                orders=orders_in_book,
                message=message  # Pass through the message
            )


    def transition_to_matching(self, order: Order, notes: Optional[str] = None):
        """Transition order to MATCHING state"""
        self.order_repository.transition_state(
            order.order_id, 
            OrderState.MATCHING,
            notes=notes
        )

    def transition_to_limit_matching(self, order: Order, notes: Optional[str] = None):
        """Transition order to LIMIT_MATCHING state"""
        self.order_repository.transition_state(
            order.order_id, 
            OrderState.LIMIT_MATCHING,
            notes=notes
        )

    def transition_to_pending(self, order: Order, notes: Optional[str] = None):
        """Transition order to PENDING state"""
        self.order_repository.transition_state(
            order.order_id, 
            OrderState.PENDING,
            notes=notes
        )

    def transition_to_active(self, order: Order, notes: Optional[str] = None):
        """Transition order to ACTIVE state"""
        self.order_repository.transition_state(
            order.order_id, 
            OrderState.ACTIVE,
            notes=notes
        )



    def transition_non_crossing_limit(self, order: Order) -> None:
        """Handle the state sequence for non-crossing limit orders"""
        # Start from COMMITTED state
        if order.state == OrderState.COMMITTED:
            self.transition_to_matching(
                order,
                notes="Starting limit order sequence"
            )
            self.transition_to_limit_matching(
                order,
                notes="Moving to limit matching"
            )
            self.transition_to_pending(
                order,
                notes="Ready to add to book"
            )
        elif order.state != OrderState.PENDING:
            self.logger.error(
                f"Invalid state for non-crossing limit order: {order.state}\n"
                f"Order history:\n{order.print_history()}"
            )
            raise ValueError(f"Invalid state for non-crossing limit order: {order.state}")

    def transition_to_validated(self, order: Order, notes: Optional[str] = None):
        """Transition order to VALIDATED state"""
        if order.state == OrderState.INPUT:  # Only transition from INPUT state
            self.order_repository.transition_state(
                order.order_id, 
                OrderState.VALIDATED,
                notes=notes or "Order validated"
            )

    def sync_agent_orders(self, agent_id: str):
        """Sync agent's orders using repository"""
        active_orders = self.order_repository.get_active_orders_from_agent(agent_id)
        self.agent_repository.sync_agent_orders(agent_id, active_orders)

