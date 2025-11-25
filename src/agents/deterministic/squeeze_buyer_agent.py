from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails

class SqueezeBuyerAgent(BaseAgent):
    """Dormant buyer that activates at specific round to create price spike (short squeeze)"""

    def __init__(self,
                 activation_round: int = 3,  # Round to start aggressive buying
                 buy_proportion: float = 0.95,  # Use 95% of cash when activated
                 price_aggression: float = 1.80,  # Bid 80% above market when activated (SQUEEZE!)
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_round = activation_round
        self.buy_proportion = buy_proportion
        self.price_aggression = price_aggression

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        current_price = market_state['price']

        # Stay dormant until activation round
        if round_number < self.activation_round:
            return TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning=f"Dormant until round {self.activation_round}",
                valuation=current_price,
                valuation_reasoning="Waiting for activation",
                price_target=current_price,
                price_target_reasoning="No target yet"
            )

        # SQUEEZE MODE: Aggressive buying to spike price
        available_cash = self.available_cash
        aggressive_price = current_price * self.price_aggression

        # Calculate massive order size
        max_shares = int(available_cash * self.buy_proportion / aggressive_price)

        if max_shares == 0:
            return TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning="Insufficient cash for squeeze order",
                valuation=current_price,
                valuation_reasoning="No cash available",
                price_target=current_price,
                price_target_reasoning="No cash available"
            )

        # Place single massive aggressive limit order
        orders = [
            OrderDetails(
                decision="Buy",
                quantity=max_shares,
                order_type=OrderType.LIMIT,
                price_limit=aggressive_price
            )
        ]

        return TradeDecision(
            orders=orders,
            replace_decision="Replace",
            reasoning=f"SQUEEZE ACTIVATED (Round {round_number}): Buying {max_shares} shares at ${aggressive_price:.2f} (+{(self.price_aggression-1)*100:.0f}% above market)",
            valuation=aggressive_price * 1.5,
            valuation_reasoning=f"Aggressive valuation to create price spike at round {round_number}",
            price_target=aggressive_price * 2.0,
            price_target_reasoning="Target: Force short squeeze by driving price up"
        )
