from typing import List
from market.orders.order import Order, OrderState

def has_commitments(order: Order) -> bool:
    """Check if order has any actual resource commitments"""
    return (
        order.current_cash_commitment > 0 or 
        order.current_share_commitment > 0
    )

def get_orders_with_commitments(orders: List[Order]) -> List[Order]:
    """Extract orders that have actual resource commitments"""
    return [
        order for order in orders 
        if has_commitments(order)
    ]

def get_active_orders(orders: List[Order]) -> List[Order]:
    """Extract active orders from a list of orders"""
    return [
        order for order in orders 
        if is_active(order)
    ]

def get_book_orders(orders: List[Order]) -> List[Order]:
    """Extract active orders from a list of orders"""
    return [
        order for order in orders 
        if is_in_book(order)
    ]

def is_active(order: Order) -> bool:
    """Check if order is active in the order book"""
    return (
        order.state in {
            OrderState.ACTIVE,
            OrderState.PENDING,
            OrderState.PARTIALLY_FILLED,
            OrderState.COMMITTED
        }
    )

def is_in_book(order: Order) -> bool:
    """Check if an order is currently in the order book"""
    return order.state in {
        OrderState.ACTIVE,
        OrderState.PARTIALLY_FILLED
    }