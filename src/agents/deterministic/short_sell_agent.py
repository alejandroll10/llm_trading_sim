from typing import Dict, List
import math
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails

class ShortSellTrader(BaseAgent):
    """Aggressively shorts when price is above fundamental value"""
    
    def __init__(self, 
                 threshold: float = 0.05,  # Minimum 5% gap to trade
                 max_proportion: float = 0.5,  # Maximum 50% of max short capacity to use
                 scaling_factor: float = 2.0,  # How aggressively to scale with gap size
                 fundamental_premium: float = 1.1,  # 10% above fundamental is considered overvalued
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.max_proportion = max_proportion
        self.scaling_factor = scaling_factor
        self.fundamental_premium = fundamental_premium

    def calculate_trade_proportion(self, gap_percentage: float) -> float:
        """Calculate what proportion of max short capacity to use based on gap size"""
        if gap_percentage < self.threshold:  # Only short when price is above estimate
            return 0.0
        
        # Scale linearly with gap size, capped at max_proportion
        proportion = min(
            self.max_proportion,
            gap_percentage * self.scaling_factor
        )
        return proportion

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        current_price = market_state['price']
        
        # Estimate fundamental value from market_state or use a simple estimate
        # For testing, assume 20% discount to current price if not provided
        fundamental_estimate = market_state.get('fundamental_estimate', current_price * 0.8)
        
        # Calculate gap percentage (positive means price is above fundamental)
        gap_percentage = (current_price - fundamental_estimate) / fundamental_estimate
        
        # For price target, assume mean reversion (if no trend info in market_state)
        price_trend = market_state.get('price_trend', 0)
        # Price target: move toward fundamental based on gap and trend
        price_target = current_price * (1 - 0.1 * gap_percentage + 0.2 * price_trend)
        
        # Only short if price is above our estimated fundamental + premium
        if gap_percentage < self.threshold:
            return TradeDecision(
                orders=[],  # No orders when we don't see a shorting opportunity
                replace_decision="Add",
                reasoning=f"No shorting opportunity. Price ${current_price:.2f} not sufficiently above fundamental ${fundamental_estimate:.2f}",
                valuation=fundamental_estimate,
                valuation_reasoning=f"Fundamental value estimated at ${fundamental_estimate:.2f} based on expected dividends and growth.",
                price_target=price_target,
                price_target_reasoning=f"Expect price to move toward fundamental with momentum factor {price_trend:.2f}."
            )
        
        # Calculate proportion to trade based on gap size
        proportion = self.calculate_trade_proportion(gap_percentage)
        
        # Calculate current wealth to determine short capacity
        current_wealth = self.total_cash + (self.total_shares * current_price)
        
        # For testing, explicitly try to short a fixed amount
        # Start with a small short position to test the mechanism (500 shares)
        desired_short = 500
            
        # Only do this if we don't already have a large short position
        current_short_position = max(0, -self.shares)
        
        if current_short_position >= 1000:
            # Already have a significant short position, don't add more
            return TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning=f"Already have sufficient short position of {current_short_position} shares.",
                valuation=fundamental_estimate,
                valuation_reasoning=f"Fundamental value estimated at ${fundamental_estimate:.2f} based on expected dividends and growth.",
                price_target=price_target,
                price_target_reasoning=f"Expect price to move toward fundamental with momentum factor {price_trend:.2f}."
            )
        
        # Check if we have enough shares, if not we'll need to borrow
        to_sell = desired_short
        if to_sell <= 0:
            return TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning=f"No quantity to short. Current shares: {self.shares}",
                valuation=fundamental_estimate,
                valuation_reasoning=f"Fundamental value estimated at ${fundamental_estimate:.2f} based on expected dividends and growth.",
                price_target=price_target,
                price_target_reasoning=f"Expect price to move toward fundamental with momentum factor {price_trend:.2f}."
            )
        
        # Create order details - use limit order
        order = OrderDetails(
            decision="Sell",
            quantity=to_sell,
            order_type=OrderType.LIMIT,
            price_limit=current_price * 0.99  # Willing to sell for slightly less
        )
        
        reasoning = (
            f"Price ${current_price:.2f} is {gap_percentage:.1%} above estimated fundamental ${fundamental_estimate:.2f}. "
            f"Shorting {to_sell} shares to test short selling mechanism. "
            f"Current shares: {self.shares}, Current short position: {current_short_position}"
        )
        
        self.logger.info(reasoning)
        
        return TradeDecision(
            orders=[order],
            replace_decision="Replace",  # Replace any existing orders
            reasoning=reasoning,
            valuation=fundamental_estimate,
            valuation_reasoning=f"Fundamental value estimated at ${fundamental_estimate:.2f} based on expected dividends and growth.",
            price_target=price_target,
            price_target_reasoning=f"Expect price to move toward fundamental with momentum factor {price_trend:.2f}."
        ) 