"""Simple multi-stock sell agent for testing"""

from agents.agents_api import OrderDetails, OrderType
from agents.deterministic.multi_stock_base import MultiStockAgent


class MultiStockSellAgent(MultiStockAgent):
    """Always sells across all stocks"""

    VALUATION_REASONING = "Simple sell strategy"

    def decide_orders(self, stocks_data: dict, round_number: int):
        orders = []

        for stock_id, stock_state in stocks_data.items():
            price = stock_state['price']
            position = self.positions.get(stock_id, 0)

            # Sell if we have shares
            if position > 50:
                sell_qty = 50
                orders.append(OrderDetails(
                    stock_id=stock_id,
                    decision="Sell",
                    quantity=sell_qty,
                    order_type=OrderType.LIMIT,
                    price_limit=price * 0.99  # Willing to sell for 1% less
                ))

        return orders, f"Placed {len(orders)} sell orders"
