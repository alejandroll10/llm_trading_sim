from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails

class SellTrader(BaseAgent):
    """Always Sells"""
    
    def __init__(self, 
                 threshold: float = 0.05,  # Minimum 5% gap to trade
                 max_proportion: float = 0.5,  # Maximum 50% of holdings/cash to trade
                 scaling_factor: float = 2.0,  # How aggressively to scale with gap size
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.max_proportion = max_proportion
        self.scaling_factor = scaling_factor

    def calculate_trade_proportion(self, gap_percentage: float) -> float:
        """Calculate what proportion of holdings/cash to trade based on gap size"""
        if abs(gap_percentage) < self.threshold:
            return 0.0
        
        # Scale linearly with gap size, capped at max_proportion
        proportion = min(
            self.max_proportion,
            abs(gap_percentage) * self.scaling_factor
        )
        return proportion

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        price = market_state['price']        
        # Calculate proportion to trade
        proportion = self.max_proportion
        
        # Always sell
        available_shares = self.available_shares
        quantity = int(available_shares * proportion)
        
        if quantity == 0:
            return TradeDecision(
                orders=[],  # Empty list for no orders
                replace_decision="Add",
                reasoning="Insufficient shares for minimum trade"
            )
        
        # Create order details
        order = OrderDetails(
            decision="Sell",
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price_limit=price * 0.99  # Willing to sell for 1% less
        )
        
        return TradeDecision(
            orders=[order],  # List with single order
            replace_decision="Replace",  # Replace any existing orders
            reasoning=f"Price ${price:.2f}, always sell",
            valuation=price * 0.9,  # Example: value it 10% below current price
            valuation_reasoning="Basic sell valuation based on current market conditions",
            price_target=price * 0.95,  # Example: expect price to drop 5%
            price_target_reasoning="Target price determined by expected market movement"
        )