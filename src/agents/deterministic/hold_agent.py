from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision

class HoldTrader(BaseAgent):
    """Always Holds - Used primarily for testing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        current_price = market_state.get('price', 0)
        decision = TradeDecision(
            orders=[],
            replace_decision="Add",
            reasoning="Always hold strategy",
            valuation=current_price,
            valuation_reasoning="Using current price as valuation baseline",
            price_prediction_reasoning="No expected price change",
            price_prediction_t=current_price,
            price_prediction_t1=current_price,
            price_prediction_t2=current_price,
        )
        self.broadcast_message(round_number, {
            'valuation': decision.valuation,
            'price_prediction_t1': decision.price_prediction_t1,
            'reasoning': decision.reasoning,
        })
        return decision
