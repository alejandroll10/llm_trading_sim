from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType

class MomentumTrader(BaseAgent):
    """Trades based on price momentum/trend following"""
    
    def __init__(self,
                 short_window: int = 5,    # Short-term moving average window
                 long_window: int = 20,    # Long-term moving average window
                 min_trend: float = 0.02,  # Minimum 2% trend to trade
                 max_position: float = 0.5, # Maximum 50% of capital in position
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.min_trend = min_trend
        self.max_position = max_position
    
    def calculate_moving_average(self, history: List, window: int) -> float:
        """Calculate simple moving average for given window size"""
        if len(history) < window:
            return history[-1]['price'] if history else 0
        
        recent_prices = [h['price'] for h in history[-window:]]
        return sum(recent_prices) / len(recent_prices)

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> Dict:
        if len(history) < self.long_window:  # Need sufficient history
            return TradeDecision(
                decision="Hold",
                quantity=0,
                reasoning="Insufficient history for momentum analysis"
            ).model_dump()

        current_price = market_state['price']
        short_ma = self.calculate_moving_average(history, self.short_window)
        long_ma = self.calculate_moving_average(history, self.long_window)
        
        # Calculate trend strength
        trend = (short_ma - long_ma) / long_ma
        
        # If trend is too weak, hold
        if abs(trend) < self.min_trend:
            return TradeDecision(
                decision="Hold",
                quantity=0,
                reasoning="Insufficient trend strength"
            ).model_dump()
        
        # Upward trend -> Buy
        if trend > 0:
            available_cash = self.available_cash
            max_shares = int(available_cash / current_price)
            quantity = int(max_shares * min(abs(trend), self.max_position))
            
            if quantity == 0:
                return TradeDecision(
                    decision="Hold",
                    quantity=0,
                    reasoning="Insufficient cash for momentum trade"
                ).model_dump()
                
            return TradeDecision(
                decision="Buy",
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price_limit=current_price * 1.01,  # Pay up to 1% more
                reasoning=f"Upward trend: Short MA ${short_ma:.2f} above Long MA ${long_ma:.2f} by {trend:.1%}"
            ).model_dump()
            
        # Downward trend -> Sell
        else:
            available_shares = self.available_shares
            quantity = int(available_shares * min(abs(trend), self.max_position))
            
            if quantity == 0:
                return TradeDecision(
                    decision="Hold",
                    quantity=0,
                    reasoning="Insufficient shares for momentum trade"
                ).model_dump()
                
            return TradeDecision(
                decision="Sell",
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price_limit=current_price * 0.99,  # Accept up to 1% less
                reasoning=f"Downward trend: Short MA ${short_ma:.2f} below Long MA ${long_ma:.2f} by {abs(trend):.1%}"
            ).model_dump()