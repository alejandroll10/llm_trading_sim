"""
Market Buy Agent - Uses MARKET orders to trigger crossed market scenarios.

This agent places MARKET buy orders, which when they can't fully fill
due to resource constraints, get converted to aggressive limit orders.
This is used to test the crossed market fix (Issue #88).
"""

from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails


class MarketBuyAgent(BaseAgent):
    """
    Deterministic agent that places MARKET buy orders.

    Used to test crossed market fix - when market orders can't fill,
    they become aggressive limits at 110% of best ask, which could
    previously cause crossed markets.
    """

    def __init__(self,
                 buy_proportion: float = 0.8,  # Use 80% of available cash
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buy_proportion = buy_proportion

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        price = market_state['price']

        # Calculate how many shares we can buy
        available_cash = self.available_cash
        max_shares = int(available_cash / price)
        quantity = int(max_shares * self.buy_proportion)

        if quantity == 0:
            return TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning="Insufficient cash for trade",
                valuation=price,
                valuation_reasoning="No trade - insufficient cash",
                price_prediction_reasoning="N/A",
                price_prediction_t=price,
                price_prediction_t1=price,
                price_prediction_t2=price,
            )

        # Place a MARKET buy order
        order = OrderDetails(
            decision="Buy",
            quantity=quantity,
            order_type=OrderType.MARKET,  # MARKET order - key for triggering the bug
            price_limit=None
        )

        return TradeDecision(
            orders=[order],
            replace_decision="Replace",
            reasoning=f"Market buy {quantity} shares at current price ${price:.2f}",
            valuation=price,
            valuation_reasoning="Market buy agent",
            price_prediction_reasoning="N/A",
            price_prediction_t=price,
            price_prediction_t1=price,
            price_prediction_t2=price,
        )
