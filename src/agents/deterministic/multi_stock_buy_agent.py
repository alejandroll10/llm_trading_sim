"""Simple multi-stock buy agent for testing"""

from agents.agents_api import OrderDetails, OrderType
from agents.deterministic.multi_stock_base import MultiStockAgent


class MultiStockBuyAgent(MultiStockAgent):
    """Always buys across all stocks"""

    VALUATION_REASONING = "Simple buy strategy"

    def decide_orders(self, stocks_data: dict, round_number: int):
        orders = []
        remaining_cash = self.cash

        for stock_id, stock_state in stocks_data.items():
            price = stock_state['price']

            # Buy if we have cash
            if remaining_cash > price * 50:
                buy_qty = 50
                cost = price * buy_qty * 1.01  # Account for price limit
                remaining_cash -= cost
                orders.append(OrderDetails(
                    stock_id=stock_id,
                    decision="Buy",
                    quantity=buy_qty,
                    order_type=OrderType.LIMIT,
                    price_limit=price * 1.01  # Willing to pay 1% more
                ))

        return orders, f"Placed {len(orders)} buy orders"
