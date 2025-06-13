from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails

class BuyToCloseTrader(BaseAgent):
    """Buys to close short positions when price falls sufficiently"""
    
    def __init__(self, 
                 threshold: float = -0.05,  # Minimum 5% drop to close positions
                 max_proportion: float = 0.7,  # Maximum 70% of short position to close
                 scaling_factor: float = 2.0,  # How aggressively to scale with gap size
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.max_proportion = max_proportion
        self.scaling_factor = scaling_factor
        self.last_price = None

    def calculate_trade_proportion(self, gap_percentage: float) -> float:
        """Calculate what proportion of short position to close based on price drop"""
        if gap_percentage > self.threshold:  # Only close when price has fallen enough
            return 0.0
        
        # Scale linearly with gap size, capped at max_proportion
        proportion = min(
            self.max_proportion,
            abs(gap_percentage) * self.scaling_factor
        )
        return proportion

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        current_price = market_state['price']
        
        # Estimate fundamental for API compliance
        fundamental_estimate = market_state.get('fundamental_estimate', current_price * 0.9)
        
        # Calculate price change relative to previous decision
        price_change_pct = 0
        if self.last_price is not None:
            price_change_pct = (current_price - self.last_price) / self.last_price
        
        self.last_price = current_price
        
        # For price target, expect continued momentum
        price_target = current_price * (1 + price_change_pct * 0.5)  # Damped continuation
        
        # Check if we have a short position (negative shares)
        current_short = max(0, -self.shares)  # A positive number representing short position
        
        if current_short == 0:
            return TradeDecision(
                orders=[],  # No short position to close
                replace_decision="Add",
                reasoning="No short position to close",
                valuation=fundamental_estimate,
                valuation_reasoning=f"Fundamental value estimated at ${fundamental_estimate:.2f}.",
                price_target=price_target,
                price_target_reasoning=f"Expect price to continue recent trend of {price_change_pct:.1%} with damping."
            )
        
        # Only close if price has dropped enough
        if price_change_pct > self.threshold:
            return TradeDecision(
                orders=[],  # Don't close position yet
                replace_decision="Add",
                reasoning=f"Price change of {price_change_pct:.1%} not below threshold of {self.threshold:.1%}",
                valuation=fundamental_estimate,
                valuation_reasoning=f"Fundamental value estimated at ${fundamental_estimate:.2f}.",
                price_target=price_target,
                price_target_reasoning=f"Expect price to continue recent trend of {price_change_pct:.1%} with damping."
            )
        
        # Calculate proportion of short position to close
        proportion = self.calculate_trade_proportion(price_change_pct)
        shares_to_buy = int(current_short * proportion)
        
        if shares_to_buy == 0:
            return TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning=f"Calculated position to close is too small",
                valuation=fundamental_estimate,
                valuation_reasoning=f"Fundamental value estimated at ${fundamental_estimate:.2f}.",
                price_target=price_target,
                price_target_reasoning=f"Expect price to continue recent trend of {price_change_pct:.1%} with damping."
            )
        
        # Create order to buy back shares
        order = OrderDetails(
            decision="Buy",
            quantity=shares_to_buy,
            order_type=OrderType.LIMIT,
            price_limit=current_price * 1.01  # Willing to pay slightly more to close
        )
        
        return TradeDecision(
            orders=[order],
            replace_decision="Replace",  # Replace any existing orders
            reasoning=f"Price dropped {price_change_pct:.1%}, closing {shares_to_buy} of {current_short} short position",
            valuation=fundamental_estimate,
            valuation_reasoning=f"Fundamental value estimated at ${fundamental_estimate:.2f}.",
            price_target=price_target,
            price_target_reasoning=f"Expect price to continue recent trend of {price_change_pct:.1%} with damping."
        ) 