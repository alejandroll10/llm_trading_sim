from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails

class MarketMakerBuy(BaseAgent):
    """Market maker that places buy orders based on fundamental value"""
    
    def __init__(self,
                 discount: float = 0.01,  # 1% discount to market price
                 max_discount: float = 0.05,   # 5% maximum discount
                 fundamental_factor: float = 1.0,  # How much to increase discount based on fundamental
                 position_limit: int = 1000000,   # Increased to match initial position
                 order_size: int = 100,        # Size of each order
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.max_discount = max_discount
        self.fundamental_factor = fundamental_factor
        self.position_limit = position_limit
        self.order_size = order_size

    def calculate_bid_price(self, price: float, fundamental: float) -> float:
        """Calculate bid price based on market price and fundamental value"""
        # Calculate percentage difference from fundamental
        fundamental_gap = (fundamental - price) / price
        
        # Increase discount when price is below fundamental (good buying opportunity)
        # Decrease discount when price is above fundamental (more cautious buying)
        adjusted_discount = self.discount + (fundamental_gap * self.fundamental_factor)
        final_discount = min(max(-self.max_discount, adjusted_discount), self.max_discount)
        
        return price * (1 - final_discount)

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        price = market_state['price']
        fundamental = market_state.get('fundamental_price', price)
        current_position = self.shares

        # Don't buy if at position limit
        if current_position >= self.position_limit:
            return TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning="At position limit",
                valuation=fundamental,
                valuation_reasoning="Fundamental estimate provided",
                price_target=price,
                price_target_reasoning="No trade executed",
            )

        # Calculate bid price
        bid_price = self.calculate_bid_price(price, fundamental)

        # Check available cash
        available_cash = self.available_cash
        max_shares = int(available_cash / bid_price)

        if max_shares == 0:
            return TradeDecision(
                orders=[],
                replace_decision="Add",
                reasoning="Insufficient cash for trade",
                valuation=fundamental,
                valuation_reasoning="Fundamental estimate provided",
                price_target=price,
                price_target_reasoning="Cannot participate",
            )

        # Adjust order size based on remaining position capacity
        buy_size = min(
            self.order_size,
            self.position_limit - current_position,
            max_shares
        )

        order = OrderDetails(
            decision="Buy",
            quantity=buy_size,
            order_type=OrderType.LIMIT,
            price_limit=bid_price,
        )
        return TradeDecision(
            orders=[order],
            replace_decision="Replace",
            reasoning=f"Making market: Bid ${bid_price:.2f} (fundamental: ${fundamental:.2f})",
            valuation=fundamental,
            valuation_reasoning="Fundamental estimate provided",
            price_target=fundamental,
            price_target_reasoning="Target aligns with fundamental",
        )
