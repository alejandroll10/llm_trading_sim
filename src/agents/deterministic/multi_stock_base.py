"""Shared base for deterministic multi-stock agents.

Every multi_stock_* agent used to repeat the same scaffolding: check
market_state['is_multi_stock'], iterate market_state['stocks'], and wrap the
resulting orders in a TradeDecision whose valuation/prediction fields are
neutral fillers (single-stock valuations are not applicable to a portfolio
strategy). This base centralizes that scaffolding; subclasses implement only
the per-portfolio order logic.
"""
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision


class MultiStockAgent(BaseAgent):
    """Template for rule-based agents that trade a multi-stock market state.

    Subclasses set VALUATION_REASONING and implement
    decide_orders(stocks_data, round_number) -> (orders, reasoning), where
    stocks_data is the per-stock state dict (market_state['stocks']).
    """

    VALUATION_REASONING = "Deterministic multi-stock strategy"
    PRICE_PREDICTION_REASONING = "N/A"

    def decide_orders(self, stocks_data: dict, round_number: int):
        """Return (list[OrderDetails], reasoning str) for this round."""
        raise NotImplementedError

    def make_decision(self, market_state: dict, history: list, round_number: int) -> TradeDecision:
        orders, reasoning = [], "No orders (not in multi-stock mode)"
        if market_state.get('is_multi_stock'):
            orders, reasoning = self.decide_orders(market_state['stocks'], round_number)
        return self.neutral_decision(orders, reasoning)

    def neutral_decision(self, orders, reasoning, valuation_reasoning=None) -> TradeDecision:
        """Wrap orders in a TradeDecision with neutral filler fields."""
        return TradeDecision(
            valuation_reasoning=valuation_reasoning or self.VALUATION_REASONING,
            valuation=0.0,
            price_prediction_reasoning=self.PRICE_PREDICTION_REASONING,
            price_prediction_t=0.0,
            price_prediction_t1=0.0,
            price_prediction_t2=0.0,
            orders=orders,
            reasoning=reasoning,
            replace_decision="Replace",
        )
