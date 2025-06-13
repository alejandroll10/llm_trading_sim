from typing import Dict, List, Optional
from market.orders.order import Order, OrderState
import logging
from agents.agent_manager.services.order_services import get_active_orders, get_book_orders

class OrderRepository:
    """Centralized repository for all orders in the system"""
    def __init__(self, logger=None):
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.state_index: Dict[OrderState, Dict[str, List[str]]] = {
            state: {'buy': [], 'sell': []} for state in OrderState
        }
        self.agent_index: Dict[str, List[str]] = {}  # agent_id -> [order_ids]
        self.logger = logger or logging.getLogger('orders')

    def create_order(self, order: Order) -> str:
        """Register a new order in the repository"""
        self.orders[order.order_id] = order
        self._index_order(order)
        
        # Use order's string representation
        self.logger.info(f"Order created: {order}")
        return order.order_id

    def transition_state(self, order_id: str, new_state: OrderState, 
                        filled_qty: float = 0, price: Optional[float] = None, 
                        notes: Optional[str] = None) -> None:
        """Updates order state with proper indexing and history tracking does not release cash or shares"""
        order = self.get_order(order_id)
        old_state = order.state
        
        # Validate transition
        if not self._is_valid_transition(old_state, new_state):
            raise ValueError(f"Invalid transition: {old_state} -> {new_state}")
        
        # Update indexes
        self.state_index[old_state][order.side].remove(order_id)
        self.state_index[new_state][order.side].append(order_id)
        
        # Update order state
        order.state = new_state
        
        # Record history with current commitment values
        order.add_history_entry(old_state, new_state, filled_qty, price, notes)
        
        # Enhanced logging with trade details
        log_msg = f"State change {old_state} → {new_state} | {order}"
        if filled_qty > 0:
            log_msg += (
                f" | Filled: {filled_qty} @ ${price:.2f}"
                f" | Total filled: {order.filled_quantity}/{order.quantity}"
                f" | Committed cash: ${order.current_cash_commitment:.2f}"
                f" | Released cash: ${order.released_cash:.2f}"
                f" | Committed shares: {order.current_share_commitment}"
                f" | Released shares: {order.released_shares}"
            )
        if notes:
            log_msg += f" | Notes: {notes}"
        
        self.logger.info(log_msg)

    def get_orders_by_state(self, state: OrderState, side: Optional[str] = None) -> List[Order]:
        """Get all orders in a given state"""
        if side:
            return [self.get_order(oid) for oid in self.state_index[state][side]]
        return [self.get_order(oid) for oid in self.state_index[state]['buy'] + self.state_index[state]['sell']]

    def get_agent_orders(self, agent_id: str) -> List[Order]:
        """Get all orders for a specific agent"""
        return [self.get_order(oid) for oid in self.agent_index.get(agent_id, [])]
    
    def _index_order(self, order: Order) -> None:
        """Update internal indexes for the order"""
        self.state_index[order.state][order.side].append(order.order_id)
        if order.agent_id not in self.agent_index:
            self.agent_index[order.agent_id] = []
        self.agent_index[order.agent_id].append(order.order_id)

    @staticmethod
    def _is_valid_transition(from_state: OrderState, to_state: OrderState) -> bool:
        """Validate state transitions"""
        valid_transitions = {
            OrderState.INPUT: {OrderState.VALIDATED, OrderState.CANCELLED},
            OrderState.VALIDATED: {OrderState.COMMITTED, OrderState.CANCELLED},
            OrderState.COMMITTED: {OrderState.MATCHING, OrderState.LIMIT_MATCHING, OrderState.PENDING, OrderState.CANCELLED},
            OrderState.MATCHING: {OrderState.PENDING, OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.LIMIT_MATCHING, OrderState.COMMITTED},
            OrderState.LIMIT_MATCHING: {OrderState.PENDING, OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED},
            OrderState.PENDING: {OrderState.ACTIVE, OrderState.CANCELLED},
            OrderState.ACTIVE: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED},
            OrderState.PARTIALLY_FILLED: {OrderState.PENDING, OrderState.FILLED, OrderState.COMMITTED, OrderState.CANCELLED},
            OrderState.FILLED: set(),  # Terminal state
            OrderState.CANCELLED: set()  # Terminal state
        }
        return to_state in valid_transitions[from_state]

    def get_order(self, order_id: str) -> Order:
        """Get a specific order by ID"""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        return self.orders[order_id]

    def print_all_orders(self, logger=None) -> str:
        """Print all orders in the repository with their history
        Args:
            logger: Optional logger object. If None, prints to terminal instead
        Returns:
            str: String representation of all orders
        """
        output = []
        output.append("\n=== ALL ORDERS IN REPOSITORY ===")
        
        # Group orders by state
        for state in OrderState:
            buy_orders = self.get_orders_by_state(state, 'buy')
            sell_orders = self.get_orders_by_state(state, 'sell')
            
            if buy_orders or sell_orders:
                output.append(f"\n--- {state.value.upper()} ---")
                
                if buy_orders:
                    output.append("BUY ORDERS:")
                    for order in buy_orders:
                        output.append(str(order))
                        output.append(order.print_history())
                        output.append("")  # Empty line for readability
                
                if sell_orders:
                    output.append("SELL ORDERS:")
                    for order in sell_orders:
                        output.append(str(order))
                        output.append(order.print_history())
                        output.append("")  # Empty line for readability

        result = "\n".join(output)
        
        if logger:
            for line in output:
                logger.info(line)
        else:
            print(result)
            
        return result
    
    def get_active_orders_from_agent(self, agent_id: str) -> List[Order]:
        """Get only active/pending/partially filled orders"""
        orders = self.get_agent_orders(agent_id)
        return get_active_orders(orders)
    
    def get_book_orders_from_agent(self, agent_id: str) -> List[Order]:
        """Get only active/pending/partially filled orders"""
        orders = self.get_agent_orders(agent_id)
        return get_book_orders(orders)

