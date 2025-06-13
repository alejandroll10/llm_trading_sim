import random
from typing import List, Tuple
from market.orders.order import Order

class OrderProcessingService:
    def __init__(self, order_book):
        self.order_book = order_book

    def split_orders_by_type(self, orders: List[Order]) -> Tuple[List[Order], List[Order]]:
        """Split orders into market and limit orders"""
        random.shuffle(orders)  # Randomize processing order
        market_orders = [o for o in orders if o.order_type == 'market']
        limit_orders = [o for o in orders if o.order_type == 'limit']
        return market_orders, limit_orders

