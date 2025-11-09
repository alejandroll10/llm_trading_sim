from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from market.orders.order import OrderId

@dataclass
class Trade:
    """
    Represents a completed trade between two agents
    """
    buyer_id: str
    seller_id: str
    stock_id: str
    quantity: float
    price: float
    timestamp: datetime
    round: int
    buyer_order_id: OrderId
    seller_order_id: OrderId
    
    def __post_init__(self):
        """Validate trade data and compute additional fields"""
        if self.quantity <= 0:
            raise ValueError("Trade quantity must be positive")
        if self.price <= 0:
            raise ValueError("Trade price must be positive")
            
        self.value = self.quantity * self.price
        
    def to_dict(self):
        """Convert trade to dictionary for logging/storage"""
        return {
            'buyer_id': self.buyer_id,
            'seller_id': self.seller_id,
            'stock_id': self.stock_id,
            'quantity': self.quantity,
            'price': self.price,
            'value': self.value,
            'timestamp': self.timestamp,
            'buyer_order_id': self.buyer_order_id,
            'seller_order_id': self.seller_order_id,
            'round': self.round
        }
        
    def __str__(self):
        """String representation for logging"""
        return (
            f"Trade: {self.stock_id} {self.quantity} @ ${self.price:.2f}"
            f", Order IDs: "
            f"{self.buyer_order_id}, {self.seller_order_id}"
            f" (Buyer: {self.buyer_id}, Seller: {self.seller_id})"
            f" Round: {self.round}"
        )

    @classmethod
    def from_orders(cls, buy_order, sell_order, quantity: float, price: float, round: int):
        """Create a trade from matching orders"""
        # Verify both orders are for the same stock
        if buy_order.stock_id != sell_order.stock_id:
            raise ValueError(
                f"Cannot create trade: buy order is for {buy_order.stock_id} "
                f"but sell order is for {sell_order.stock_id}"
            )

        return cls(
            buyer_id=buy_order.agent_id,
            seller_id=sell_order.agent_id,
            stock_id=buy_order.stock_id,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            buyer_order_id=buy_order.order_id,
            seller_order_id=sell_order.order_id,
            round=round
        )