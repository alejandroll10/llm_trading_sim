"""Simple multi-stock buy agent for testing"""

from agents.base_agent import BaseAgent
from typing import List


class MultiStockBuyAgent(BaseAgent):
    """Always buys across all stocks"""

    def make_decision(self, market_state: dict, history: list, round_number: int):
        from agents.agents_api import TradeDecision, OrderDetails, OrderType

        orders = []

        if market_state.get('is_multi_stock'):
            stocks_data = market_state['stocks']
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

        return TradeDecision(
            valuation_reasoning="Simple buy strategy",
            valuation=0.0,
            price_target_reasoning="N/A",
            price_target=0.0,
            orders=orders,
            reasoning=f"Placed {len(orders)} buy orders",
            replace_decision="Replace"
        )
