"""Multi-stock market maker agent for testing - provides liquidity on both sides"""

from agents.agents_api import OrderDetails, OrderType
from agents.deterministic.multi_stock_base import MultiStockAgent


class MultiStockMarketMaker(MultiStockAgent):
    """Provides buy and sell liquidity across all stocks in multi-stock mode"""

    VALUATION_REASONING = "Market maker providing liquidity"

    def __init__(self, spread_pct: float = 0.02, order_size: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spread_pct = spread_pct  # 2% spread
        self.order_size = order_size

    def decide_orders(self, stocks_data: dict, round_number: int):
        orders = []
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

        return orders, f"Market maker placed {len(orders)} orders for liquidity"
