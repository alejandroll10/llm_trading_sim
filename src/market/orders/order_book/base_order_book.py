from typing import Optional, List
from market.orders.order import Order
from market.state.sim_context import SimulationContext
from datetime import datetime

class OrderBook:
    def __init__(self, context: SimulationContext, order_repository, logger=None):
        self.context = context
        self.logger = logger
        self.order_repository = order_repository
        # Own our state
        self._buy_orders = []
        self._sell_orders = []
        
    @property
    def buy_orders(self):
        """Access buy orders"""
        return self._buy_orders
    
    @buy_orders.setter
    def buy_orders(self, value):
        """Set buy orders and update public view"""
        self._buy_orders = value
        self._update_public_view()
    
    @property
    def sell_orders(self):
        """Access sell orders"""
        return self._sell_orders
    
    @sell_orders.setter
    def sell_orders(self, value):
        """Set sell orders and update public view"""
        self._sell_orders = value
        self._update_public_view()

    def _update_public_view(self):
        """Update the public view in context"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        aggregated_levels = self.get_aggregated_levels()
        
        # Create current quote snapshot
        current_quote = {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'timestamp': datetime.now().isoformat(),
            'round': self.context.round_number
        }
        
        # Update current order book state
        self.context.public_info['order_book_state'].update({
            'best_bid': best_bid,
            'best_ask': best_ask,
            'midpoint': self.get_midpoint(),
            'aggregated_levels': aggregated_levels
        })
        
        # Add to market history only
        self.context.market_history.quote_history.append(current_quote)

    # These methods remain unchanged as they work with our internal state
    def is_empty(self):
        return len(self._buy_orders) == 0 and len(self._sell_orders) == 0

    def is_empty_for_side(self, side: str) -> bool:
        if side == 'buy':
            return len(self._sell_orders) == 0
        elif side == 'sell':
            return len(self._buy_orders) == 0
        raise ValueError("Side must be 'buy' or 'sell'")

    def contains_order(self, order: Order) -> bool:
        orders = self._buy_orders if order.side == 'buy' else self._sell_orders
        return any(entry.order.order_id == order.order_id for entry in orders)

    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        for entry in self._buy_orders + self._sell_orders:
            if entry.order.order_id == order_id:
                return entry.order
        return None

    def get_agent_orders(self, agent_id: str) -> dict[str, List[Order]]:
        agent_orders = {
            'buy': [entry.order for entry in self._buy_orders 
                   if entry.order.agent_id == agent_id],
            'sell': [entry.order for entry in self._sell_orders 
                    if entry.order.agent_id == agent_id]
        }
        return agent_orders