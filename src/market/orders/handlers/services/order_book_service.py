from dataclasses import dataclass
from market.orders.order import Order
from services.logging_service import LoggingService

@dataclass
class BookAddResult:
    success: bool
    message: str
    would_cross: bool
    crossing_price: float = 0

class LimitOrderBookService:
    def __init__(self, order_book, order_state_manager, logger=None):
        self._order_book = order_book
        self.order_state_manager = order_state_manager
        
    def add_to_book(self, order: Order) -> BookAddResult:
        """Add order to book if it doesn't cross"""
        if self._would_cross_market(order):
            crossing_price = self._get_crossing_price(order)
            return BookAddResult(
                success=False,
                message=f"Order would cross market at {crossing_price}",
                would_cross=True,
                crossing_price=crossing_price
            )
        
        try:
            # First transition through states
            self.order_state_manager.transition_non_crossing_limit(order)
            
            # Now that order is in PENDING state, add to book
            self._order_book.add_limit_order(order)
            
            # Important: Transition to ACTIVE immediately after adding to book
            self.order_state_manager.transition_to_active(
                order,
                notes="Order successfully added to book"
            )
            
            return BookAddResult(
                success=True,
                message="Order added to book and activated",
                would_cross=False
            )
        except ValueError as e:
            LoggingService.get_logger('order_book').error(f"Failed to add order to book: {str(e)}")
            return BookAddResult(
                success=False,
                message=str(e),
                would_cross=False
            )
        
    def _would_cross_market(self, order: Order) -> bool:
        """Check if a limit order would cross with existing orders"""
        if order.side == 'buy':
            best_ask = self._order_book.peek_best_sell()
            return best_ask is not None and order.price >= best_ask.price
        else:  # sell order
            best_bid = self._order_book.peek_best_buy()
            return best_bid is not None and order.price <= best_bid.price
            
    def _get_crossing_price(self, order: Order) -> float:
        """Get the price at which the order would cross"""
        if order.side == 'buy':
            best_ask = self._order_book.peek_best_sell()
            return best_ask.price if best_ask else 0
        else:
            best_bid = self._order_book.peek_best_buy()
            return abs(best_bid.price) if best_bid else 0