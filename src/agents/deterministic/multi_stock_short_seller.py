"""Multi-stock short seller agent for testing margin calls across stocks"""

from agents.agents_api import OrderDetails, OrderType
from agents.deterministic.multi_stock_base import MultiStockAgent


class MultiStockShortSeller(MultiStockAgent):
    """Shorts overvalued stocks in multi-stock mode to build short positions for margin testing.

    Like the single-stock ShortSellTrader, this agent aggressively builds short positions
    to test margin call mechanics. It sells 500 shares per stock regardless of current holdings,
    which forces borrowing from the lending pool.
    """

    VALUATION_REASONING = "Short overvalued stocks to build positions for margin testing"

    def __init__(self, target_short_per_stock: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_short_per_stock = target_short_per_stock
        self.has_built_position = False  # Only build position once

    def decide_orders(self, stocks_data: dict, round_number: int):
        orders = []

        for stock_id, stock_state in stocks_data.items():
            price = stock_state['price']

            # Get current borrowed position for this stock
            current_borrowed = self.borrowed_positions.get(stock_id, 0)

            # Build short position if we haven't hit target
            # Sell 500 shares total - if we own 100, we'll borrow 400
            if current_borrowed < self.target_short_per_stock and not self.has_built_position:
                # Sell aggressively - the system will borrow what we don't own
                short_qty = self.target_short_per_stock

                orders.append(OrderDetails(
                    stock_id=stock_id,
                    decision="Sell",
                    quantity=short_qty,
                    order_type=OrderType.LIMIT,
                    price_limit=price * 0.99  # Willing to sell for 1% less
                ))

        if orders:
            self.has_built_position = True

        return orders, f"Placed {len(orders)} short sell orders across stocks"
