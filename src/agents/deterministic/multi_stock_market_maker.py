"""Multi-stock market maker agent for testing - provides liquidity on both sides"""

from agents.base_agent import BaseAgent
from typing import List


class MultiStockMarketMaker(BaseAgent):
    """Provides buy and sell liquidity across all stocks in multi-stock mode"""

    def __init__(self, spread_pct: float = 0.02, order_size: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spread_pct = spread_pct  # 2% spread
        self.order_size = order_size

    def make_decision(self, market_state: dict, history: list, round_number: int):
        from agents.agents_api import TradeDecision, OrderDetails, OrderType

        orders = []

        if market_state.get('is_multi_stock'):
            stocks_data = market_state['stocks']
            num_stocks = len(stocks_data)
            cash_per_stock = self.cash / max(num_stocks, 1)

            for stock_id, stock_state in stocks_data.items():
                price = stock_state['price']
                position = self.positions.get(stock_id, 0)

                # Calculate prices
                bid_price = price * (1 - self.spread_pct / 2)  # Buy at 1% below
                ask_price = price * (1 + self.spread_pct / 2)  # Sell at 1% above

                # Buy order - provide liquidity for sellers
                max_buy = int(cash_per_stock / (price * 1.02))
                buy_qty = min(self.order_size, max_buy)
                if buy_qty > 0:
                    orders.append(OrderDetails(
                        stock_id=stock_id,
                        decision="Buy",
                        quantity=buy_qty,
                        order_type=OrderType.LIMIT,
                        price_limit=bid_price
                    ))

                # Sell order - provide liquidity for buyers
                if position > 0:
                    sell_qty = min(self.order_size, position)
                    orders.append(OrderDetails(
                        stock_id=stock_id,
                        decision="Sell",
                        quantity=sell_qty,
                        order_type=OrderType.LIMIT,
                        price_limit=ask_price
                    ))

        return TradeDecision(
            valuation_reasoning="Market maker providing liquidity",
            valuation=0.0,
            price_target_reasoning="N/A",
            price_target=0.0,
            orders=orders,
            reasoning=f"Market maker placed {len(orders)} orders for liquidity",
            replace_decision="Replace"
        )
