
from dataclasses import dataclass, field
from market.orders.order import Order

@dataclass(order=True)
class OrderEntry:
    """Represents an entry in the order book with price-time priority"""
    price: float
    timestamp: float
    order: Order = field(compare=False)  # Exclude from comparison

    @classmethod
    def create_buy(cls, order: Order) -> 'OrderEntry':
        """Create a buy order entry with negative price for correct heap ordering"""
        return cls(-abs(order.price), order.timestamp, order)
    
    @classmethod
    def create_sell(cls, order: Order) -> 'OrderEntry':
        """Create a sell order entry"""
        return cls(abs(order.price), order.timestamp, order)
    
    @property
    def display_price(self) -> float:
        """Get the actual price for display purposes"""
        return abs(self.price)

    def __post_init__(self):
        """Validate the entry after initialization"""
        if not isinstance(self.order, Order):
            raise ValueError("order must be an instance of Order")
        if self.order.price is None:
            raise ValueError("order must have a price")

    @property
    def remaining_quantity(self) -> int:
        """Get remaining quantity from the order"""
        return self.order.remaining_quantity

    @property
    def agent_id(self) -> str:
        """Get agent ID from the order"""
        return self.order.agent_id

    @property
    def quantity(self) -> int:
        """Get the total quantity from the order"""
        return self.order.quantity