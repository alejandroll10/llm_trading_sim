"""
Agent that places both market and limit orders to test commitment tracking.
This reproduces the bug where market orders in MATCHING state weren't counted as active.
"""
from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails


class MixedOrderAgent(BaseAgent):
    """Places both market sell and limit sell/buy orders each round."""

    def __init__(self,
                 market_sell_qty: int = 1000,
                 limit_sell_qty: int = 1500,
                 limit_buy_qty: int = 1500,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.market_sell_qty = market_sell_qty
        self.limit_sell_qty = limit_sell_qty
        self.limit_buy_qty = limit_buy_qty

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        price = market_state['price']
        orders = []

        # Market sell if we have shares
        if self.available_shares >= self.market_sell_qty:
            orders.append(OrderDetails(
                decision="Sell",
                quantity=self.market_sell_qty,
                order_type=OrderType.MARKET,
                price_limit=None
            ))

        # Limit sell if we have shares
        remaining_shares = self.available_shares - self.market_sell_qty
        if remaining_shares >= self.limit_sell_qty:
            orders.append(OrderDetails(
                decision="Sell",
                quantity=self.limit_sell_qty,
                order_type=OrderType.LIMIT,
                price_limit=price * 1.02  # 2% above current price
            ))

        # Limit buy if we have cash
        buy_cost = self.limit_buy_qty * price * 0.98
        if self.available_cash >= buy_cost:
            orders.append(OrderDetails(
                decision="Buy",
                quantity=self.limit_buy_qty,
                order_type=OrderType.LIMIT,
                price_limit=price * 0.98  # 2% below current price
            ))

        return TradeDecision(
            orders=orders,
            replace_decision="Replace",
            reasoning=f"Mixed order test: market sell {self.market_sell_qty}, limit sell {self.limit_sell_qty}, limit buy {self.limit_buy_qty}",
            valuation=price,
            valuation_reasoning="Test valuation",
            price_prediction_reasoning="Test prediction",
            price_prediction_t=price,
            price_prediction_t1=price,
            price_prediction_t2=price,
        )
