from abc import ABC
from market.orders.order import Order

class BaseOrderHandler(ABC):
    def __init__(self, order_book, agent_manager, order_state_manager, context):
        self.order_book = order_book
        self.agent_manager = agent_manager
        self.agent_repository = agent_manager.agent_repository
        self.order_state_manager = order_state_manager
        self.context = context

    def check_commitment_sufficient(self, order: Order, trade_cost: float) -> bool:
        """Check if order has sufficient commitment for trade"""
        if order.side == 'buy':
            return order.current_cash_commitment >= trade_cost
        return order.current_share_commitment >= trade_cost  # For sell orders, cost = quantity