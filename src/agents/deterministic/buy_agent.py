from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails

class BuyTrader(BaseAgent):
    """Always Buys"""
    
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
        # Calculate total quantity to trade
        available_cash = self.available_cash
        max_shares = int(available_cash / price)
        total_quantity = int(max_shares * self.max_proportion)
        
        if total_quantity == 0:
            return TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning="Insufficient cash for minimum trade"
            )
        
        # Split quantity between two orders
        aggressive_quantity = int(total_quantity * 0.6)  # 60% at higher price
        patient_quantity = total_quantity - aggressive_quantity  # 40% at lower price
        
        orders = [
            # More aggressive order at 1% above market
            OrderDetails(
                decision="Buy",
                quantity=aggressive_quantity,
                order_type=OrderType.LIMIT,
                price_limit=price * 1.01
            ),
            # More patient order at 1% below market
            OrderDetails(
                decision="Buy",
                quantity=patient_quantity,
                order_type=OrderType.LIMIT,
                price_limit=price * 0.99
            )
        ]
        
        return TradeDecision(
            orders=orders,
            replace_decision="Replace",  # Replace any existing orders
            reasoning=f"Price ${price:.2f}, placing orders at +1% and -1% levels",
            valuation=100.0,  # Replace with your actual valuation logic
            valuation_reasoning="Basic buy valuation based on current market conditions",
            price_prediction_reasoning="Target price determined by historical growth patterns",
            price_prediction_t=price,
            price_prediction_t1=price * 1.05,
            price_prediction_t2=price * 1.05,
        )