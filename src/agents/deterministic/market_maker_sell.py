from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails

class MarketMakerSell(BaseAgent):
    """Market maker that places sell orders based on fundamental value"""

    def __init__(self,
                 markup: float = 0.01,  # 1% markup to market price
                 max_markup: float = 0.05,   # 5% maximum markup
                 fundamental_factor: float = 1.0,  # How much to increase markup based on fundamental
                 position_limit: int = 0,   # Maximum short position (negative)
                 order_size: int = 100,        # Size of each order
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.markup = markup
        self.max_markup = max_markup
        self.fundamental_factor = fundamental_factor
        self.position_limit = position_limit
        self.order_size = order_size
        self.position_limit = 0

    def calculate_ask_price(self, price: float, fundamental: float) -> float:
        """Calculate ask price based on market price and fundamental value"""
        # Calculate percentage difference from fundamental
        fundamental_gap = (price - fundamental) / price

        # Increase markup when price is above fundamental (good selling opportunity)
        # Decrease markup when price is below fundamental (more cautious selling)
        adjusted_markup = self.markup + (fundamental_gap * self.fundamental_factor)
        final_markup = min(max(-self.max_markup, adjusted_markup), self.max_markup)

        return price * (1 + final_markup)

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        price = market_state['price']
        fundamental = market_state.get('fundamental_price', price)
        current_position = self.shares

        # Only sell if we have shares above the position limit (0 or negative)
        if current_position > self.position_limit:
            # Calculate ask price
            ask_price = self.calculate_ask_price(price, fundamental)

            # Check available shares
            available_shares = self.available_shares
            if available_shares == 0:
                return TradeDecision(
                    orders=[],
                    replace_decision="Add",
                    reasoning="Insufficient shares for trade",
                    valuation=fundamental,
                    valuation_reasoning="Fundamental estimate provided",
                    price_prediction_reasoning="Cannot participate",
                    price_prediction_t=price,
                    price_prediction_t1=price,
                    price_prediction_t2=price,
                )

            # Adjust order size based on remaining position capacity
            sell_size = min(
                self.order_size,
                current_position - self.position_limit,
                available_shares
            )

            order = OrderDetails(
                decision="Sell",
                quantity=sell_size,
                order_type=OrderType.LIMIT,
                price_limit=ask_price,
            )
            return TradeDecision(
                orders=[order],
                replace_decision="Replace",
                reasoning=f"Making market: Ask ${ask_price:.2f} (fundamental: ${fundamental:.2f})",
                valuation=fundamental,
                valuation_reasoning="Fundamental estimate provided",
                price_prediction_reasoning="Target aligns with fundamental",
                price_prediction_t=fundamental,
                price_prediction_t1=fundamental,
                price_prediction_t2=fundamental,
            )

        return TradeDecision(
            orders=[],
            replace_decision="Add",
            reasoning="At position limit",
            valuation=fundamental,
            valuation_reasoning="Fundamental estimate provided",
            price_prediction_reasoning="No trade executed",
            price_prediction_t=price,
            price_prediction_t1=price,
            price_prediction_t2=price,
        )
