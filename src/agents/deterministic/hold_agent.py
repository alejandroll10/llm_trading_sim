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
            price_target=current_price,
            price_target_reasoning="No expected price change",
        )
        self.broadcast_message(round_number, {
            'valuation': decision.valuation,
            'price_target': decision.price_target,
            'reasoning': decision.reasoning,
        })
        return decision
