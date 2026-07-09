"""
Multi-Stock Value Trader - Deterministic Agent for Testing Multi-Stock Support

Strategy:
- Sells overvalued stocks (price > fundamental)
- Buys undervalued stocks (price < fundamental)
"""

from agents.agents_api import OrderDetails, OrderType
from agents.deterministic.multi_stock_base import MultiStockAgent


class MultiStockValueTrader(MultiStockAgent):
    """
    A simple multi-stock value trader.

    Sells overvalued stocks, buys undervalued stocks based on fundamentals.
    """

    VALUATION_REASONING = "Value-based strategy across multiple stocks"
    PRICE_PREDICTION_REASONING = "Target based on fundamentals"

    def decide_orders(self, stocks_data: dict, round_number: int):
        orders = []
        reasoning_parts = []
        remaining_cash = self.cash

        for stock_id, stock_state in stocks_data.items():
            price = stock_state['price']
            fundamental = stock_state['fundamental_price']
            position = self.positions.get(stock_id, 0)

            if self.logger:
                self.logger.info(
                    f"Agent {self.agent_id} analyzing {stock_id}: "
                    f"Price=${price:.2f}, Fundamental=${fundamental:.2f}, Position={position}"
                )

            # Simple value trading logic
            if price > fundamental * 1.02 and position > 100:
                # Overvalued: sell some shares
                sell_qty = min(100, position)
                orders.append(OrderDetails(
                    stock_id=stock_id,
                    decision="Sell",
                    quantity=sell_qty,
                    order_type=OrderType.LIMIT,
                    price_limit=price
                ))
                reasoning_parts.append(
                    f"{stock_id}: SELL {sell_qty} @ ${price:.2f} (overvalued vs ${fundamental:.2f})"
                )
            elif price < fundamental * 0.98 and remaining_cash > price * 100:
                # Undervalued: buy some shares
                buy_qty = min(100, int(remaining_cash / price))
                cost = price * buy_qty
                remaining_cash -= cost
                orders.append(OrderDetails(
                    stock_id=stock_id,
                    decision="Buy",
                    quantity=buy_qty,
                    order_type=OrderType.LIMIT,
                    price_limit=price
                ))
                reasoning_parts.append(
                    f"{stock_id}: BUY {buy_qty} @ ${price:.2f} (undervalued vs ${fundamental:.2f})"
                )
            else:
                reasoning_parts.append(f"{stock_id}: HOLD (fairly valued)")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No trading opportunities"
        return orders, reasoning
