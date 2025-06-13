from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType

class MeanReversionTrader(BaseAgent):
    """Trades based on deviations from moving average"""
    
    def __init__(self,
                 window_size: int = 10,  # Length of moving average window
                 threshold: float = 0.03,  # Minimum 3% deviation to trade
                 max_position: float = 0.5,  # Maximum 50% of capital in position
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.threshold = threshold
        self.max_position = max_position
    
    def calculate_moving_average(self, history: List) -> float:
        """Calculate simple moving average of recent prices"""
        if len(history) < self.window_size:
            return history[-1]['price'] if history else 0
        
        recent_prices = [h['price'] for h in history[-self.window_size:]]
        return sum(recent_prices) / len(recent_prices)

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> Dict:
        if len(history) < 2:  # Need at least 2 data points
            return TradeDecision(
                decision="Hold",
                quantity=0,
                reasoning="Insufficient history for mean reversion"
            ).model_dump()

        current_price = market_state['price']
        moving_avg = self.calculate_moving_average(history)
        
        # Calculate deviation as percentage
        deviation = (current_price - moving_avg) / moving_avg
        
        # If deviation is too small, hold
        if abs(deviation) < self.threshold:
            return TradeDecision(
                decision="Hold",
                quantity=0,
                reasoning="Price near moving average"
            ).model_dump()
        
        # Price above MA -> Sell
        if deviation > 0:
            available_shares = self.available_shares
            # Calculate position size based on deviation
            quantity = int(available_shares * min(abs(deviation), self.max_position))
            
            if quantity == 0:
                return TradeDecision(
                    decision="Hold",
                    quantity=0,
                    reasoning="Insufficient shares for mean reversion trade"
                ).model_dump()
                
            return TradeDecision(
                decision="Sell",
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price_limit=current_price * 0.99,  # Accept up to 1% less
                reasoning=f"Price ${current_price:.2f} above MA ${moving_avg:.2f} by {deviation:.1%}"
            ).model_dump()
            
        # Price below MA -> Buy
        else:
            available_cash = self.available_cash
            max_shares = int(available_cash / current_price)
            quantity = int(max_shares * min(abs(deviation), self.max_position))
            
            if quantity == 0:
                return TradeDecision(
                    decision="Hold",
                    quantity=0,
                    reasoning="Insufficient cash for mean reversion trade"
                ).model_dump()
                
            return TradeDecision(
                decision="Buy",
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price_limit=current_price * 1.01,  # Pay up to 1% more
                reasoning=f"Price ${current_price:.2f} below MA ${moving_avg:.2f} by {abs(deviation):.1%}"
            ).model_dump()