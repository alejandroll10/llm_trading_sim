import time
import uuid
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, NewType
from constants import FLOAT_TOLERANCE

# Create new ID types
OrderId = NewType('OrderId', str)
AgentId = NewType('AgentId', int)  # Assuming agent IDs are integers

class OrderState(Enum):
    INPUT = "input"         # Submitted by agents, not validated
    VALIDATED = "validated" # Passed validation checks
    COMMITTED = "committed" # Passed commitment checks
    MATCHING = "matching"   # Attempting market-to-market matching
    LIMIT_MATCHING = "limit_matching"  # Attempting limit-to-book matching
    PENDING = "pending"     # Validated, waiting to be added to book
    ACTIVE = "active"       # In order book
    PARTIALLY_FILLED = "partially_filled"  # Some quantity executed
    FILLED = "filled"       # Fully executed
    CANCELLED = "cancelled" # Removed from book

@dataclass
class OrderHistoryEntry:
    timestamp: float
    round_placed: int
    side: str
    from_state: Optional[OrderState]
    to_state: OrderState
    filled_quantity: float = 0
    price: Optional[float] = None
    notes: Optional[str] = None

class Order:
    def __init__(self, agent_id, order_type, side, quantity, round_placed, stock_id=None, price=None, timestamp=None, order_id=None, replace_decision=None, is_margin_call=False):
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
        if order_type not in ['market', 'limit']:
            raise ValueError(f"Invalid order_type: {order_type}. Must be 'market' or 'limit'")
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}. Must be positive")
        if order_type == 'limit' and price is None:
            raise ValueError("Limit orders must specify a price")

        self.agent_id = agent_id
        self.stock_id = stock_id or "DEFAULT_STOCK"  # Default for backwards compatibility
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.round_placed = round_placed
        # Convert datetime timestamp to float if needed
        if isinstance(timestamp, datetime):
            self.timestamp = timestamp.timestamp()
        else:
            self.timestamp = timestamp or time.time()
        self.remaining_quantity = quantity
        self.order_id = order_id or str(uuid.uuid4())
        self.state = OrderState.INPUT
        self.filled_quantity = 0
        self.original_cash_commitment = 0    # Initial cash committed, never changes
        self.original_share_commitment = 0   # Initial shares committed, never changes
        self.current_cash_commitment = 0     # Current cash still committed, reduces with releases
        self.current_share_commitment = 0    # Current shares still committed, reduces with releases
        self.released_cash = 0              # Total cash released so far
        self.released_shares = 0            # Total shares released so far
        self.replace_decision = replace_decision
        self.is_margin_call = is_margin_call  # Flag for forced margin call orders
        # Let handlers manage commitments during validation
        # Don't commit anything at creation time

        self.history: List[OrderHistoryEntry] = []
        # Record initial state
        self.add_history_entry(None, OrderState.INPUT)

    @property
    def remaining_commitment(self) -> float:
        """Get remaining commitment needed"""
        if self.side == 'buy':
            return self.current_cash_commitment  # Already represents remaining commitment
        return self.current_share_commitment    # Should also represent remaining commitment

    def release_commitment(self, amount: float):
        """Release commitment for the order"""
        if self.side == 'buy':
            # Check if amount is within tolerance of available commitment
            if amount - self.current_cash_commitment > FLOAT_TOLERANCE:
                raise ValueError(
                    f"Cannot release more than committed. "
                    f"Attempted: {amount}, Available: {self.current_cash_commitment}, "
                    f"Original committed: {self.original_cash_commitment}, "
                    f"Current committed: {self.current_cash_commitment}, "
                    f"Already released: {self.released_cash}"
                )
            self.current_cash_commitment = max(0, self.current_cash_commitment - amount)
            self.released_cash += amount  # Keep for tracking purposes
        else:
            if amount - self.current_share_commitment > FLOAT_TOLERANCE:
                raise ValueError(
                    f"Cannot release more than committed. "
                    f"Attempted: {amount}, Available: {self.current_share_commitment}, "
                    f"Original committed: {self.original_share_commitment}, "
                    f"Current committed: {self.current_share_commitment}, "
                    f"Already released: {self.released_shares}"
                )
            self.current_share_commitment = max(0, self.current_share_commitment - amount)
            self.released_shares += amount  # Keep for tracking purposes

    def add_history_entry(self, from_state: Optional[OrderState], to_state: OrderState, 
                         filled_qty: float = 0, price: Optional[float] = None, 
                         notes: Optional[str] = None) -> None:
        """Add a new entry to the order history"""
        # Build commitment details
        commitment_details = []
        if self.side == 'buy':
            commitment_details.extend([
                f"cash_commit={self.current_cash_commitment:.2f}",
                f"orig_cash_commit={self.original_cash_commitment:.2f}",
                f"released_cash={self.released_cash:.2f}"
            ])
        else:  # sell
            commitment_details.extend([
                f"share_commit={self.current_share_commitment}",
                f"orig_share_commit={self.original_share_commitment}",
                f"released_shares={self.released_shares}"
            ])

        # Add commitment info to notes
        commitment_info = f"[{', '.join(commitment_details)}]"
        full_notes = f"{notes + ' ' if notes else ''}{commitment_info}"

        entry = OrderHistoryEntry(
            timestamp=time.time(),
            round_placed=self.round_placed,
            side=self.side,
            from_state=from_state,
            to_state=to_state,
            filled_quantity=filled_qty,
            price=price,
            notes=full_notes
        )
        self.history.append(entry)

    def __str__(self) -> str:
        price_str = f"${self.price:.2f}" if self.price is not None else "None"
        return (
            f"Order(id={self.order_id}, "
            f"stock={self.stock_id}, "
            f"side={self.side}, "
            f"qty={self.quantity}@{price_str}, "
            f"agent={self.agent_id}, "
            f"state={self.state}, "
            f"original_cash_commitment={self.original_cash_commitment:.2f}, "
            f"current_cash_commitment={self.current_cash_commitment:.2f}, "
            f"released_cash={self.released_cash:.2f}, "
            f"original_share_commitment={self.original_share_commitment}, "
            f"current_share_commitment={self.current_share_commitment}, "
            f"released_shares={self.released_shares})"
        )

    def print_history(self) -> str:
        """Returns a formatted string of the order's history"""
        lines = [f"History for Order {self.order_id}:"]
        for entry in self.history:
            state_change = f"{entry.from_state.value if entry.from_state else 'INIT'} → {entry.to_state.value}"
            timestamp_str = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            details = []
            if entry.filled_quantity > 0:
                details.append(f"filled: {entry.filled_quantity}")
            if entry.price is not None:
                details.append(f"price: ${entry.price:.2f}")
            if entry.notes:
                details.append(f"notes: {entry.notes}")
            
            details_str = f" ({', '.join(details)})" if details else ""
            lines.append(f"  {timestamp_str}: {state_change}{details_str}")
        
        return "\n".join(lines)