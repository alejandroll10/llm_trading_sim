from dataclasses import dataclass
import math
from typing import Optional
from market.orders.order import Order, OrderState

@dataclass
class ConversionResult:
    new_order: Optional[Order]
    original_order: Order
    success: bool
    message: str
    transferred_commitment: float

class OrderConversionService:
    def __init__(self, order_book, order_repository, logger):
        self._order_book = order_book
        self._order_repository = order_repository
        self._logger = logger
        
    def convert_to_aggressive_limit(self, order: Order, current_price: float) -> ConversionResult:
        """Convert market order to aggressive limit order"""
        # Calculate new aggressive price
        new_price = self._calculate_aggressive_price(order, current_price)
        
        # For buy orders, verify commitment at new price
        if order.side == 'buy':
            max_quantity = math.floor(order.current_cash_commitment / new_price)
            if max_quantity < order.remaining_quantity:
                self._logger.warning(
                    f"Reducing order quantity from {order.remaining_quantity} from agent {order.agent_id} to {max_quantity}. Original order: {order.print_history()} "
                    f"due to higher aggressive price"
                )
                order.remaining_quantity = max_quantity
                if max_quantity == 0:
                    return ConversionResult(
                        new_order=None,
                        original_order=order,
                        success=False,
                        message="Zero quantity after price adjustment",
                        transferred_commitment=0
                    )
        
        # Update order price
        order.price = new_price
        
        # Transition to COMMITTED
        self._order_repository.transition_state(
            order.order_id, 
            OrderState.COMMITTED,
            notes=f"Converted to aggressive limit @ ${new_price:.2f}, not yet added to order book"
        )
        
        return ConversionResult(
            new_order=order,  # Return same order
            original_order=order,
            success=True,
            message="Successfully converted to aggressive limit, not yet added to order book",
            transferred_commitment=order.current_cash_commitment  # No transfer needed
        )
    
    def _calculate_aggressive_price(self, order: Order, current_price: float) -> float:
        """Calculate aggressive price based on order side and market conditions"""
        if order.side == 'buy':
            best_ask = self._order_book.get_best_ask() or current_price
            return best_ask * 1.10  # 10% above best ask
        else:
            best_bid = self._order_book.get_best_bid() or current_price
            return best_bid * 0.90  # 10% below best bid
            
    def _transfer_commitments(self, old_order: Order, new_order: Order, commitment: float):
        """Transfer commitments from old order to new order"""
        if old_order.side == 'buy':
            new_order.current_cash_commitment = commitment
            old_order.current_cash_commitment = 0
        else:
            new_order.current_share_commitment = commitment
            old_order.current_share_commitment = 0