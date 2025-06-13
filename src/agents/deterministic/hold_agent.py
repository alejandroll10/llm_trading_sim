from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision

class HoldTrader(BaseAgent):
    """Always Holds - Used primarily for testing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> Dict:
        return TradeDecision(
            decision="Hold",
            quantity=0,
            reasoning="Always hold strategy"
        ).model_dump()