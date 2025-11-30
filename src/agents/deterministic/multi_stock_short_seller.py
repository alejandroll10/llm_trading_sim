"""Multi-stock short seller agent for testing margin calls across stocks"""

from agents.base_agent import BaseAgent
from typing import List


class MultiStockShortSeller(BaseAgent):
    """Shorts overvalued stocks in multi-stock mode to build short positions for margin testing.

    Like the single-stock ShortSellTrader, this agent aggressively builds short positions
    to test margin call mechanics. It sells 500 shares per stock regardless of current holdings,
    which forces borrowing from the lending pool.
    """

    def __init__(self, target_short_per_stock: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_short_per_stock = target_short_per_stock
        self.has_built_position = False  # Only build position once

    def make_decision(self, market_state: dict, history: list, round_number: int):
        from agents.agents_api import TradeDecision, OrderDetails, OrderType

        orders = []

        if market_state.get('is_multi_stock'):
            stocks_data = market_state['stocks']

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

        return TradeDecision(
            valuation_reasoning="Short overvalued stocks to build positions for margin testing",
            valuation=0.0,
            price_prediction_reasoning="N/A",
            price_prediction_t=0.0,
            price_prediction_t1=0.0,
            price_prediction_t2=0.0,
            orders=orders,
            reasoning=f"Placed {len(orders)} short sell orders across stocks",
            replace_decision="Replace"
        )
