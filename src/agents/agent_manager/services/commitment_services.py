from market.trade import Trade
from market.orders.order import Order


def release_for_cancellation(agent_repository, logger, order: Order):
    """Release all remaining commitments"""

    if not order.remaining_commitment > 0:
        logger.error(
            f"No commitment to release for order {order.order_id}\n"
            f"Order details: {order}\n"
            f"Attempted to release {order.remaining_commitment} {order.side} commitment\n"
            f"Order history: {order.print_history()}"
        )
        raise ValueError(f"No commitment to release for order {order.order_id}")

    
    release_commitment_agent(
        agent_repository,
        logger,
        order, 
        order.remaining_commitment,
        'cash' if order.side == 'buy' else 'shares'
    )

def release_commitment_agent(agent_repository, logger, order: Order, amount: float,
                             resource_type: str, return_borrowed: bool = True):
    """Helper for releasing commitments"""
    try:
        result = agent_repository.release_resources(
            order.agent_id,
            cash_amount=amount if resource_type == 'cash' else 0,
            share_amount=amount if resource_type == 'shares' else 0,
            return_borrowed=return_borrowed,
            stock_id=order.stock_id  # Pass stock_id for multi-stock support
        )
        
        if not result.success:
            raise ValueError(f"Failed to release resources in release_commitment_agent\n:\n {result.message}"
                             f"Attempted to release {amount} {resource_type} for order {order.order_id}")
            
        order.release_commitment(amount)
        
        logger.info(
            f"Released {amount:.2f} {resource_type} for order {order.order_id}"
        )
        
    except ValueError as e:
        # Add detailed debug information when release fails
        logger.error(
            f"Failed to release commitment:\n {str(e)}\n"
            f"Order details: {order}\n"
            f"Order history: {order.print_history()}"
        )
        raise

def release_for_trade(trade: Trade, order_repository, agent_repository, commitment_calculator, logger):
    """Release commitments after trade execution"""
    buy_order = order_repository.get_order(trade.buyer_order_id)
    sell_order = order_repository.get_order(trade.seller_order_id)

    # Calculate release amounts
    cash_to_release = commitment_calculator.calculate_release_amount(trade, buy_order)
    shares_to_release = commitment_calculator.calculate_release_amount(trade, sell_order)

    # Release through repository
    release_commitment_agent(agent_repository, logger, buy_order, cash_to_release, 'cash')
    release_commitment_agent(agent_repository, logger, sell_order, shares_to_release, 'shares', return_borrowed=False)


class CommitmentCalculator:
    """Handles all commitment-related calculations"""
    def __init__(self, order_book=None, order_books=None):
        """Initialize commitment calculator

        Args:
            order_book: Single order book (single-stock mode or backwards compatibility)
            order_books: Dict of {stock_id: OrderBook} for multi-stock mode
        """
        self.order_book = order_book
        self.order_books = order_books  # For multi-stock support

    def calculate_required_commitment(self, order: Order, current_price: float) -> float:
        """Calculate initial commitment required"""
        if order.side == 'sell':
            return order.quantity

        if order.order_type == 'market':
            return self._calculate_market_buy_commitment(order, current_price)
        else:
            return self._calculate_limit_buy_commitment(order)

    def _calculate_limit_buy_commitment(self, order: Order) -> float:
        """Simple limit order commitment"""
        return order.quantity * order.price
    
    @staticmethod
    def calculate_release_amount(trade: Trade, order: Order) -> float:
        """Calculate how much of the committed (held) funds/shares to release after a trade"""
        if order.side == 'buy':
            if order.order_type == 'market':
                # Market order logic unchanged
                proportion = trade.quantity / (order.remaining_quantity + trade.quantity)
                release_amount = order.current_cash_commitment * proportion
                
                if order.remaining_quantity == 0:
                    release_amount = order.current_cash_commitment
            else:
                # For limit orders:
                # 1. Release the amount used for the trade
                release_amount = trade.quantity * trade.price
                
                # 2. If order is now complete, also release any remaining commitment
                #    (this handles cases where we got a better price than limit)
                if order.remaining_quantity == 0:
                    release_amount = order.current_cash_commitment
                
            return min(release_amount, order.current_cash_commitment)
        else:
            return min(trade.quantity, order.current_share_commitment)

    def _calculate_market_buy_commitment(self, order: Order, current_price: float) -> float:
        """Dynamic market order commitment based on order book"""
        # Get the correct order book for this stock
        order_book = self._get_order_book_for_stock(order.stock_id)

        if not order_book:
            # Fallback if no order book available
            return order.quantity * (current_price * 1.1)  # 10% buffer

        total_cost, fillable_qty = order_book.estimate_market_order_cost(
            order.quantity, 'buy'
        )

        if fillable_qty < order.quantity:
            unfilled = order.quantity - fillable_qty
            total_cost += unfilled * (current_price * 1.1)  # 10% buffer

        return total_cost

    def _get_order_book_for_stock(self, stock_id: str):
        """Get the appropriate order book for a given stock

        Args:
            stock_id: Stock identifier (e.g., 'TECH_A' or 'DEFAULT_STOCK')

        Returns:
            OrderBook for the specified stock, or fallback order book
        """
        if self.order_books:
            # Multi-stock mode: look up specific order book
            return self.order_books.get(stock_id, self.order_book)
        else:
            # Single-stock mode: use the single order book
            return self.order_book

    def calculate_sell_commitment(self, order: Order) -> float:
        """All sell orders commit full quantity"""
        return order.quantity