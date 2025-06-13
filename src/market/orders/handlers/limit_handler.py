from typing import List
from market.orders.order import Order, OrderState
from market.trade import Trade
import logging
from market.orders.handlers.base_handler import BaseOrderHandler
from market.orders.handlers.services.order_book_service import LimitOrderBookService
from market.orders.handlers.services.limit_matching_service import LimitMatchingService
from market.orders.handlers.services.sync_service import AgentSyncService

class LimitOrderHandler(BaseOrderHandler):
    """
    Handles all limit order processing including:
    1. Matching limit orders against the order book
    2. Managing the order book for limit orders
    """

    def __init__(self, *, order_book, agent_manager, trade_execution_service, context, logger=None, 
                 order_repository=None, order_state_manager=None):
        super().__init__(order_book=order_book, 
                         agent_manager=agent_manager, 
                         order_state_manager=order_state_manager,
                         context=context)
        self.logger = logger or logging.getLogger(__name__)
        self.order_repository = order_repository
        self.order_state_manager = order_state_manager
        
        # Initialize servicess
        self._matching_service = LimitMatchingService(
            order_book=order_book,
            order_state_manager=order_state_manager,
            trade_execution_service=trade_execution_service,
            logger=logger,
            context=self.context
        )
        
        self._book_service = LimitOrderBookService(
            order_book=order_book,
            order_state_manager=order_state_manager,
            logger=logger
        )
        
        self._sync_service = AgentSyncService(
            agent_repository=agent_manager.agent_repository,
            order_repository=order_repository,
            logger=logger
        )
    
    def process_orders(self, limit_orders: List[Order]) -> List[Trade]:
        """Process orders that are ready for matching"""
        trades = []
        
        for order in limit_orders:
            try:
                # Handle state transitions for matching
                if order.state == OrderState.COMMITTED:
                    self.order_state_manager.transition_to_matching(
                        order,
                        notes="Starting limit order matching"
                    )
                    self.order_state_manager.transition_to_limit_matching(
                        order,
                        notes="Ready for limit matching"
                    )
                
                # Match order using service
                match_result = self._matching_service.match_limit_order(order)
                
                if match_result.error:
                    self.logger.error(f"Error matching order {order.order_id}: {match_result.error}")
                    continue
                    
                trades.extend(match_result.trades)
                
                # Handle remaining quantity - ensure order is ACTIVE before adding to book
                if not match_result.is_fully_matched:
                    self.order_state_manager.transition_to_pending(
                        order,
                        notes="Moving to pending for book addition"
                    )
                    book_result = self._book_service.add_to_book(order)
                    # Note: add_to_book now handles the transition to ACTIVE
                
                # Sync agent orders
                self._sync_service.sync_agent_orders_from_order_repository(order.agent_id)
                
            except ValueError as e:
                self.logger.error(
                    f"Failed to process order {order.order_id}: {str(e)}\n"
                    f"Order history:\n{order.print_history()}"
                )
                continue
                
        return trades

    def add_non_crossing_orders(self, limit_orders: List[Order]) -> List[Order]:
        """Separate crossing from non-crossing orders and handle book additions"""
        crossing_orders = []
        
        for order in limit_orders:
            # Use book service to check crossing and handle state transitions
            book_result = self._book_service.add_to_book(order)
            
            if book_result.would_cross:
                crossing_orders.append(order)
                self.logger.debug(
                    f"Order {order.order_id} would cross at {book_result.crossing_price}"
                )
            elif book_result.success:
                # Order was successfully added to book and transitioned to ACTIVE
                self.logger.info(
                    f"Added non-crossing {order.side} order to book: {order.quantity} @ "
                    f"${abs(order.price):.2f} (Agent {order.agent_id})"
                )
            else:
                # Order failed to be added for some other reason
                self.logger.error(
                    f"Failed to add order to book: {book_result.message}\n"
                    f"Order history:\n{order.print_history()}"
                )
            
            # Sync agent orders after any state changes
            self._sync_service.sync_agent_orders_from_order_repository(order.agent_id)
        
        return crossing_orders