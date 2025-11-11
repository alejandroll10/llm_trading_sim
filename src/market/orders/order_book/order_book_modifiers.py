from market.orders.order import Order, OrderState
from market.orders.order_entry import OrderEntry
from typing import List, Optional
import heapq
from services.logging_service import LoggingService

class OrderBookModifiers:
    
    def _push_order(self, entry: OrderEntry, side: str):
        """Safe way to push orders that maintains heap invariant"""
        valid_states = [OrderState.PENDING, OrderState.PARTIALLY_FILLED]
        if entry.order.state not in valid_states:
            raise ValueError(
                f"Cannot push order in state: {entry.order.state}. "
                f"Valid states are: {', '.join(str(s) for s in valid_states)}"
            )
            
        if side == 'buy':
            orders = self.buy_orders.copy()
            heapq.heappush(orders, entry)
            self.buy_orders = orders
        else:
            orders = self.sell_orders.copy()
            heapq.heappush(orders, entry)
            self.sell_orders = orders

    def clear(self):
        """Clear all orders from the book"""
        self.buy_orders = []
        self.sell_orders = []
        self._update_public_view()

    def add_limit_order(self, order: Order) -> None:
        """Add a limit order to the book"""
        LoggingService.log_order_state(f"Adding limit order: {order.side.upper()} {order.quantity} @ ${order.price:.2f}")
        
        self._validate_limit_order(order)
        entry = OrderEntry.create_buy(order) if order.side == 'buy' else OrderEntry.create_sell(order)
        
        self._push_order(entry, order.side)
        self._update_public_view()

    def pop_best_buy(self) -> Optional[OrderEntry]:
        """Remove and return the best buy order entry"""
        if not self.buy_orders:
            return None
        entry = heapq.heappop(self.buy_orders)
        # Note: Don't transition state here as it will be handled by the order handler
        self._update_public_view()
        return entry
    def pop_best_sell(self) -> Optional[OrderEntry]:
        """Remove and return the best sell order entry"""
        if not self.sell_orders:
            return None
        entry = heapq.heappop(self.sell_orders)
        self._update_public_view()
        return entry
    
    def remove_agent_orders(self, agent_id: str) -> tuple[List[OrderEntry], List[OrderEntry]]:
        """Remove all orders for a specific agent with state updates"""
        if self.logger:
            self.logger.info(f"Removing orders for Agent {agent_id}")
        
        # Store existing orders before removal
        existing_buys = [entry for entry in self.buy_orders if entry.agent_id == agent_id]
        existing_sells = [entry for entry in self.sell_orders if entry.agent_id == agent_id]
        
        # Log removals
        if existing_buys or existing_sells:
            self._log_removals(existing_buys, existing_sells)
        
        # Remove orders and update states
        self.buy_orders = [entry for entry in self.buy_orders if entry.agent_id != agent_id]
        self.sell_orders = [entry for entry in self.sell_orders if entry.agent_id != agent_id]
        
        # Update states for removed orders
        for entry in existing_buys + existing_sells:
            if entry.order.state in [OrderState.ACTIVE, OrderState.PARTIALLY_FILLED]:
                self.order_repository.transition_state(entry.order.order_id, OrderState.CANCELLED)
        
        self._update_public_view()
        return existing_buys, existing_sells
    
    def push_buy(self, entry: OrderEntry) -> None:
        """Push a buy order entry to the book"""
        self._push_order(entry, 'buy')

    def push_sell(self, entry: OrderEntry) -> None:
        """Push a sell order entry to the book"""
        self._push_order(entry, 'sell')

    def _remove_order_from_book(self, order: Order) -> None:
        """Remove an order from the book and update its state"""
        if order.state in [OrderState.ACTIVE, OrderState.PARTIALLY_FILLED]:
            self.order_repository.transition_state(order.order_id, OrderState.CANCELLED)
        
        # Remove from appropriate side
        if order.side == 'buy':
            self.buy_orders = [entry for entry in self.buy_orders 
                              if entry.order.order_id != order.order_id]
        else:
            self.sell_orders = [entry for entry in self.sell_orders 
                              if entry.order.order_id != order.order_id]
        
        self._update_public_view()

    def _validate_limit_order(self, order: Order) -> None:
        """Validate order before adding to book"""
        if not order.price:
            raise ValueError("Limit order must have a price")
        if order.state != OrderState.PENDING:
            self.logger.error(f"Order validation failed. Current state: {order.state}")
            self.logger.error(f"Order history:\n{order.print_history()}")
            raise ValueError(
                f"Invalid order state for book addition: {order.state}. "
                f"Order must be in PENDING state before adding to book."
            )
        if order.remaining_quantity <= 0:
            raise ValueError("Order must have positive remaining quantity")
