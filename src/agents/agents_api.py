from pydantic import BaseModel, model_validator
from enum import Enum
import re
from agents.agent_types import *
from typing import Literal, List
import re

class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"

class OrderDetails(BaseModel):
    """Individual order details"""
    decision: Literal["Buy", "Sell"]
    quantity: int
    order_type: OrderType
    price_limit: float | None = None
    stock_id: str = "DEFAULT_STOCK"  # NEW: Multi-stock support, defaults for backwards compatibility

    @model_validator(mode='after')
    def validate_order_details(self):
        """Validate individual order details"""
        if self.quantity <= 0:
            raise ValueError("quantity must be positive for all orders")
            
        # Convert negative quantities to positive for sell orders
        if self.decision == "Sell":
            self.quantity = abs(self.quantity)
        elif self.decision == "Buy" and self.quantity < 0:
            raise ValueError("quantity must be positive for buy orders")
            
        if self.order_type == OrderType.LIMIT and self.price_limit is None:
            raise ValueError("price_limit must be set for limit orders")
        
        if self.order_type == OrderType.MARKET:
            self.price_limit = None
            
        return self

class TradeDecision(BaseModel):
    """Decision model for trading actions.

    Fields:
        valuation_reasoning: Explanation of valuation analysis (separate from trade reasoning)
        valuation: Agent's estimated fundamental value of the asset
        price_target_reasoning: Explanation for the price target
        price_target: Agent's predicted price in the next round
        reasoning: Explanation for the trading decision (comes before orders for better chain-of-thought)
        orders: List of individual orders
        replace_decision: How to handle existing orders
            - "Add": Place new orders alongside existing ones
            - "Cancel": Cancels all existing orders
            - "Replace": Cancels all existing orders and places new orders
        message_reasoning: Reasoning for the social media message (what effect do you want it to have?)
        post_message: Optional message to post to social feed
        notes_to_self: Optional notes to your future self about what you learned this round,
            patterns observed, or strategy adjustments. These notes appear in your memory log
            in future rounds.
    """
    valuation_reasoning: str
    valuation: float
    price_target_reasoning: str
    price_target: float
    reasoning: str
    orders: List[OrderDetails] = []
    replace_decision: Literal["Cancel", "Replace", "Add"] = "Replace"
    message_reasoning: str | None = None
    post_message: str | None = None
    notes_to_self: str | None = None

    @model_validator(mode='after')
    def validate_order_api(self):
        """Validate the complete order"""

        # Handle Cancel replace_decision
        if self.replace_decision == "Cancel":
            self.orders = []
            return self

        # For Replace or Add, validate price limits from reasoning if needed
        if self.orders:
            price_matches = re.findall(r'\$?(\d+\.?\d*)', self.reasoning)
            for order in self.orders:
                if (order.order_type == OrderType.LIMIT and
                    order.price_limit is None and
                    price_matches):
                    order.price_limit = float(price_matches[0])

        return self

    model_config = {
        "coerce_numbers_to_str": False,
        "str_strip_whitespace": True,
        "str_to_lower": True,
        "from_attributes": True
    }
