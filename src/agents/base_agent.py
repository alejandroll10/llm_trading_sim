from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal, Any
import numpy as np
from market.trade import Trade
from market.information.information_types import InformationType, InformationSignal, InfoCapability
from agents.agents_api import TradeDecision
import logging
from services.logging_service import LoggingService
from services.messaging_service import MessagingService

@dataclass
class AgentType:
    name: str
    type_id: str
    system_prompt: Optional[str] = None  # Optional for deterministic agents
    user_prompt_template: Optional[str] = None  # Optional for deterministic agents

@dataclass
class Payment:
    round_number: int
    amount: float
    account: str
    payment_type: Literal['interest', 'dividend', 'trade', 'borrow_fee', 'redemption', 'other']
    stock_id: Optional[str] = None  # Optional stock identifier for multi-stock scenarios

class BaseAgent(ABC):
    """Base agent with core functionality"""
    def __init__(self, agent_id: str, initial_cash: float = 0, 
                 initial_shares: int = 0, position_limit: int = None, 
                 allow_short_selling: bool = False, 
                 margin_requirement: float = 0.5,  # 50% margin requirement by default
                 margin_base: str = "cash",  # "cash" or "wealth"
                 logger=None, info_signals_logger=None, initial_price: float = np.nan):
        self.agent_id = agent_id
        # Store initial values
        self.initial_cash = initial_cash
        self.initial_shares = initial_shares
        self.initial_dividend_cash = 0.0
        self.initial_commited_cash = 0.0

        # Current values
        self.cash = initial_cash
        # Create a basic AgentType for deterministic agents
        self.agent_type = AgentType(
            name=self.__class__.__name__,  # Use class name as default
            type_id=self.__class__.__name__.lower(),  # Lowercase version as ID
        )
        self.dividend_cash = 0.0  # Separate account for dividends/interests

        # Multi-stock positions: Dict[stock_id, shares]
        # For backwards compatibility, convert single initial_shares to dict
        self.positions = {"DEFAULT_STOCK": initial_shares}
        self.committed_positions = {"DEFAULT_STOCK": 0}
        self.borrowed_positions = {"DEFAULT_STOCK": 0}

        self.position_limit = position_limit
        self.allow_short_selling = allow_short_selling
        self.margin_requirement = margin_requirement  # Percentage of base required as margin
        self.margin_base = margin_base  # "cash" or "wealth"
        self.order_history = []
        self.decision_history = []
        self.trade_history: List[Trade] = []
        self.trade_stats = {
            'buys': 0,
            'sells': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'buy_value': 0.0,
            'sell_value': 0.0,
            'avg_buy_price': 0.0,
            'avg_sell_price': 0.0
        }
        
        # Initialize orders dictionary with all possible states
        self.orders = {
            # Pre-book states
            'input': {'buy': [], 'sell': []},
            'validated': {'buy': [], 'sell': []},
            'matching': {'buy': [], 'sell': []},
            'limit_matching': {'buy': [], 'sell': []},
            
            # Active states
            'pending': {'buy': [], 'sell': []},
            'active': {'buy': [], 'sell': []},
            'partially_filled': {'buy': [], 'sell': []},
            
            # Terminal states
            'filled': {'buy': [], 'sell': []},
            'cancelled': {'buy': [], 'sell': []}
        }
        
        # Initialize outstanding orders (active orders only)
        self.outstanding_orders = {
            'buy': [],
            'sell': []
        }
        
        self.committed_cash = 0
        self.committed_shares = 0
        self.logger = logger
        self.info_signals_logger = info_signals_logger
        self.wealth = self.cash + self.shares * initial_price + self.dividend_cash
        self.info_capabilities: Dict[InformationType, InfoCapability] = {}
        self.private_signals: Dict[InformationType, InformationSignal] = {}
        self.signal_history: Dict[int, Dict[InformationType, InformationSignal]] = {}
        self.last_update_round: int = 0
        self.last_replace_decision: Literal["Cancel", "Replace", "Add"] = "Replace"
        self.payment_history: Dict[str, List[Payment]] = {
            'interest': [],
            'dividend': [],
            'trade': [],
            'borrow_fee': [],
            'redemption': [],
            'other': []
        }

    # Backwards compatibility properties for single-stock code
    @property
    def shares(self) -> int:
        """Get shares for DEFAULT_STOCK (backwards compatibility)"""
        return self.positions.get("DEFAULT_STOCK", 0)

    @shares.setter
    def shares(self, value: int):
        """Set shares for DEFAULT_STOCK (backwards compatibility)"""
        self.positions["DEFAULT_STOCK"] = value

    @property
    def committed_shares(self) -> int:
        """Get committed shares for DEFAULT_STOCK (backwards compatibility)"""
        return self.committed_positions.get("DEFAULT_STOCK", 0)

    @committed_shares.setter
    def committed_shares(self, value: int):
        """Set committed shares for DEFAULT_STOCK (backwards compatibility)"""
        self.committed_positions["DEFAULT_STOCK"] = value

    @property
    def borrowed_shares(self) -> int:
        """Get borrowed shares for DEFAULT_STOCK (backwards compatibility)"""
        return self.borrowed_positions.get("DEFAULT_STOCK", 0)

    @borrowed_shares.setter
    def borrowed_shares(self, value: int):
        """Set borrowed shares for DEFAULT_STOCK (backwards compatibility)"""
        self.borrowed_positions["DEFAULT_STOCK"] = value

    @property
    def total_borrowed_shares(self) -> int:
        """Get total borrowed shares across all stocks

        Note: DEFAULT_STOCK acts as an accumulator for multi-stock scenarios,
        so we only sum non-DEFAULT stocks OR only DEFAULT if that's all there is.
        """
        # Get all borrowed positions
        all_positions = self.borrowed_positions

        # If we only have DEFAULT_STOCK (or nothing), use its value
        if len(all_positions) <= 1:
            return all_positions.get("DEFAULT_STOCK", 0)

        # Multi-stock scenario: sum all NON-DEFAULT stocks
        # (DEFAULT_STOCK is an accumulator to avoid double-counting)
        return sum(
            shares for stock_id, shares in all_positions.items()
            if stock_id != "DEFAULT_STOCK"
        )

    def _check_borrowed_positions_invariants(self) -> bool:
        """Verify borrowed positions invariants are maintained

        This defensive check ensures internal consistency of the borrowed_positions
        tracking system, particularly the DEFAULT_STOCK accumulator pattern.

        Invariants checked:
        1. DEFAULT_STOCK accumulator matches sum of other stocks (multi-stock mode)
        2. No borrowed positions are negative
        3. total_borrowed_shares is consistent with individual positions

        Returns:
            bool: True if all invariants pass

        Raises:
            AssertionError: If any invariant is violated (only in debug mode)
        """
        all_positions = self.borrowed_positions

        # Invariant 1: No negative borrowed positions
        for stock_id, shares in all_positions.items():
            if shares < 0:
                error_msg = f"INVARIANT VIOLATION: Negative borrowed position for {stock_id}: {shares}"
                LoggingService.log_agent_state(
                    agent_id=self.agent_id,
                    operation="INVARIANT_VIOLATION",
                    amount=error_msg,
                    agent_state=self._get_state_dict(),
                    is_error=True
                )
                assert False, error_msg

        # Invariant 2: DEFAULT_STOCK accumulator consistency (multi-stock only)
        if len(all_positions) > 1:
            # Multi-stock mode: DEFAULT_STOCK should equal sum of other stocks
            default_stock_value = all_positions.get("DEFAULT_STOCK", 0)
            other_stocks_sum = sum(
                shares for stock_id, shares in all_positions.items()
                if stock_id != "DEFAULT_STOCK"
            )

            # Allow small floating point tolerance
            tolerance = 0.01
            if abs(default_stock_value - other_stocks_sum) > tolerance:
                error_msg = (
                    f"INVARIANT VIOLATION: DEFAULT_STOCK accumulator mismatch. "
                    f"DEFAULT_STOCK={default_stock_value}, sum of other stocks={other_stocks_sum}. "
                    f"All positions: {all_positions}"
                )
                LoggingService.log_agent_state(
                    agent_id=self.agent_id,
                    operation="INVARIANT_VIOLATION",
                    amount=error_msg,
                    agent_state=self._get_state_dict(),
                    is_error=True
                )
                # Warning instead of assertion failure in production
                # This allows simulation to continue while logging the issue
                import warnings
                warnings.warn(error_msg, RuntimeWarning)
                return False

        # Invariant 3: total_borrowed_shares consistency
        expected_total = self.total_borrowed_shares
        if len(all_positions) <= 1:
            # Single-stock: should equal DEFAULT_STOCK
            actual_total = all_positions.get("DEFAULT_STOCK", 0)
        else:
            # Multi-stock: should equal sum of non-DEFAULT stocks
            actual_total = sum(
                shares for stock_id, shares in all_positions.items()
                if stock_id != "DEFAULT_STOCK"
            )

        tolerance = 0.01
        if abs(expected_total - actual_total) > tolerance:
            error_msg = (
                f"INVARIANT VIOLATION: total_borrowed_shares inconsistency. "
                f"total_borrowed_shares={expected_total}, actual sum={actual_total}"
            )
            LoggingService.log_agent_state(
                agent_id=self.agent_id,
                operation="INVARIANT_VIOLATION",
                amount=error_msg,
                agent_state=self._get_state_dict(),
                is_error=True
            )
            import warnings
            warnings.warn(error_msg, RuntimeWarning)
            return False

        return True

    def get_last_round_messages(self, round_number: int):
        """Retrieve broadcast messages from the previous round."""
        if round_number <= 0:
            return []
        return MessagingService.get_messages(round_number - 1)

    def get_message_history(self, round_number: int):
        """Retrieve all broadcast messages up to the previous round."""
        if round_number <= 1:
            return []
        return MessagingService.get_message_history(round_number - 1)

    def broadcast_message(self, round_number: int, message: Dict[str, Any]):
        """Broadcast a structured message for other agents to read next round."""
        MessagingService.add_message(round_number, self.agent_id, message)

    def read_state(self):
        """Read and log the current state of the agent."""
        state = self._get_state_dict()
        LoggingService.log_agent_state(
            agent_id=self.agent_id,
            operation="read state",
            agent_state=state,
            outstanding_orders=self.outstanding_orders
        )
        return state
    
    @abstractmethod
    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        """Make trading decision based on market state and history
        
        Args:
            market_state: Current market state information
            history: Historical market data
            round_number: Current round number
        
        Returns:
            TradeDecision: Contains:
                - orders: List of OrderDetails for new orders
                - replace_decision: "Cancel", "Replace", or "Add"
                - reasoning: Explanation for the decision
        """
        pass

    def _log_agent_state(self, operation: str, amount: float = None, 
                        include_orders: bool = True, 
                        include_order_history: bool = False,
                        print_to_terminal: bool = False):
        """Helper method to log agent state consistently"""
        # Determine log level based on operation
        is_error = "failed" in operation.lower()
        log_level = logging.ERROR if is_error else logging.INFO
        
        message = [f"\n========== Agent {self.agent_id} {operation} =========="]
        
        if amount is not None:
            message.append(f"\nAmount: {amount}")
            
        message.extend([
            f"\nCurrent state:",
            f"\n - Total cash: {self.total_cash:.2f}",
            f"\n - Available cash: {self.cash:.2f}",
            f"\n - Committed cash: {self.committed_cash:.2f}",
            f"\n - Dividend cash: {self.dividend_cash:.2f}",
            f"\n - Total shares: {self.total_shares}",
            f"\n - Available shares: {self.shares}",
            f"\n - Committed shares: {self.committed_shares}"
        ])
        
        if include_orders:
            message.extend([
                f"\nOutstanding orders:",
                f"\n - Buy orders ({len(self.outstanding_orders['buy'])}):",
            ])
            for order in self.outstanding_orders['buy']:
                message.append(f"\n   * {order}")
                message.append(f"\n     State: {order.state}")
                
            message.extend([
                f"\n - Sell orders ({len(self.outstanding_orders['sell'])}):",
            ])
            for order in self.outstanding_orders['sell']:
                message.append(f"\n   * {order}")
                message.append(f"\n     State: {order.state}")

        if include_order_history:
            message.extend([
                f"\nOrder history ({len(self.order_history)}):",
            ])
            for order in self.order_history:
                message.append(f"\n * {order}")
                message.append(f"\n   History: {order.print_history()}")
        
        full_message = ''.join(message)
        
        # Log with appropriate level
        if log_level == logging.ERROR:
            self.logger.error(full_message)
        else:
            self.logger.info(full_message)
        
        if print_to_terminal:
            print(full_message)

    def commit_cash(self, amount: float, debug: bool = False):
        """Commit cash for a trade"""
        # Log initial state
        LoggingService.log_agent_state(
            agent_id=self.agent_id,
            operation="committing cash",
            amount=amount,
            agent_state=self._get_state_dict(),
            outstanding_orders=self.outstanding_orders
        )
        
        if amount > self.cash:
            # Log error state
            LoggingService.log_agent_state(
                agent_id=self.agent_id,
                operation="commit cash failed",
                amount=amount,
                agent_state=self._get_state_dict(),
                outstanding_orders=self.outstanding_orders,
                order_history=self.order_history,
                print_to_terminal=debug,
                is_error=True
            )
            
            # Log validation error
            LoggingService.log_validation_error(
                round_number=self.last_update_round,
                agent_id=self.agent_id,
                agent_type=self.agent_type.name,
                error_type="INSUFFICIENT_CASH",
                details=f"Required: {amount:.2f}, Available: {self.cash:.2f}",
                attempted_action="COMMIT_CASH"
            )
            
            raise ValueError(f"Insufficient cash: {amount} > {self.cash}")
            
        self.committed_cash += amount
        self.cash -= amount
        
        # Log final state
        LoggingService.log_agent_state(
            agent_id=self.agent_id,
            operation="after cash commit",
            agent_state=self._get_state_dict()
        )

    def _get_state_dict(self) -> dict:
        """Helper method to get current state as dictionary"""
        return {
            'total_cash': self.total_cash,
            'available_cash': self.cash,
            'committed_cash': self.committed_cash,
            'dividend_cash': self.dividend_cash,
            'total_shares': self.total_shares,
            'available_shares': self.shares,
            'committed_shares': self.committed_shares,
            'borrowed_shares': self.borrowed_shares,
            'total_borrowed_shares': self.total_borrowed_shares,
            'net_shares': self.total_shares - self.total_borrowed_shares,
            'margin_requirement': self.margin_requirement,
            'margin_base': self.margin_base
        }

    def _release_cash(self, amount: float):
        """Release committed cash"""
        LoggingService.log_agent_state(
            agent_id=self.agent_id,
            operation="releasing cash",
            amount=amount,
            agent_state=self._get_state_dict(),
            outstanding_orders=self.outstanding_orders
        )
        
        FLOAT_TOLERANCE = 1e-10  # Add tolerance for floating point comparisons
        
        if amount - self.committed_cash > FLOAT_TOLERANCE:
            raise ValueError(f"Cannot release more than committed: {amount} > {self.committed_cash}")
        
        self.committed_cash = max(0, self.committed_cash - amount)
        self.cash += amount
        
        LoggingService.log_agent_state(
            agent_id=self.agent_id,
            operation="after cash release",
            agent_state=self._get_state_dict()
        )

    def commit_shares(self, quantity: float, round_number: int, current_price: float, debug: bool = False, stock_id: str = "DEFAULT_STOCK"):
        """Commit shares for a pending order, borrowing shares if allowed and necessary

        Args:
            quantity: Number of shares to commit
            round_number: Current round number (for logging)
            current_price: Current market price (for margin calculations)
            debug: Whether to print debug information
            stock_id: Which stock's shares to commit (for multi-stock mode)
        """
        # Log initial state
        LoggingService.log_agent_state(
            agent_id=self.agent_id,
            operation=f"committing shares ({stock_id})",
            amount=quantity,
            agent_state=self._get_state_dict(),
            outstanding_orders=self.outstanding_orders
        )

        # Get current position for this specific stock
        current_shares = self.positions.get(stock_id, 0)

        # Check if we need to borrow shares
        shares_to_borrow = 0
        if quantity > current_shares:
            # Agent is trying to sell more than they have
            if not self.allow_short_selling:
                # Log error state
                LoggingService.log_agent_state(
                    agent_id=self.agent_id,
                    operation=f"commit shares failed, round {round_number}",
                    amount=quantity,
                    agent_state=self._get_state_dict(),
                    outstanding_orders=self.outstanding_orders,
                    order_history=self.order_history,
                    print_to_terminal=debug,
                    is_error=True
                )

                # Log validation error
                LoggingService.log_validation_error(
                    round_number=round_number,
                    agent_id=self.agent_id,
                    agent_type=self.agent_type.name,
                    error_type="INSUFFICIENT_SHARES",
                    details=f"Required: {quantity}, Available: {current_shares} ({stock_id})",
                    attempted_action="COMMIT_SHARES"
                )

                raise ValueError(f"Insufficient shares: required: {quantity} > available: {current_shares}")
            else:
                # Short selling is allowed, calculate shares to borrow
                shares_to_borrow = quantity - current_shares

                # Check margin requirements
                max_borrowable = self.get_max_borrowable_shares(current_price)
                total_borrowed_after = self.total_borrowed_shares + shares_to_borrow
                
                if total_borrowed_after > max_borrowable:
                    # Log error state
                    LoggingService.log_agent_state(
                        agent_id=self.agent_id,
                        operation=f"commit shares failed (margin requirement), round {round_number}",
                        amount=quantity,
                        agent_state=self._get_state_dict(),
                        outstanding_orders=self.outstanding_orders,
                        order_history=self.order_history,
                        print_to_terminal=debug,
                        is_error=True
                    )
                    
                    # Calculate margin base for the error message
                    margin_base_value = self.cash if self.margin_base == "cash" else (
                        self.total_cash + (self.total_shares - self.total_borrowed_shares) * current_price
                    )
                    
                    # Log validation error
                    LoggingService.log_validation_error(
                        round_number=round_number,
                        agent_id=self.agent_id,
                        agent_type=self.agent_type.name,
                        error_type="MARGIN_REQUIREMENT_NOT_MET",
                        details=(f"Current borrowed: {self.total_borrowed_shares}, "
                                f"Attempting to borrow: {shares_to_borrow}, "
                                f"Max allowed: {max_borrowable}, "
                                f"Margin base ({self.margin_base}): {margin_base_value:.2f}, "
                                f"Requirement: {self.margin_requirement:.2%}"),
                        attempted_action="COMMIT_SHARES"
                    )
                    
                    raise ValueError(f"Margin requirement not met: "
                                   f"Attempted to borrow {shares_to_borrow} shares, "
                                   f"but can only borrow {max_borrowable:.2f} "
                                   f"based on {self.margin_requirement:.2%} margin requirement")
                
                # All checks passed, borrow the shares
                # Track borrowed shares per stock
                current_borrowed = self.borrowed_positions.get(stock_id, 0)
                self.borrowed_positions[stock_id] = current_borrowed + shares_to_borrow
                # Also update global if not DEFAULT_STOCK (avoid double counting)
                if stock_id != "DEFAULT_STOCK":
                    self.borrowed_shares += shares_to_borrow

        # Track committed shares per stock
        current_committed = self.committed_positions.get(stock_id, 0)
        self.committed_positions[stock_id] = current_committed + quantity
        # Also update global if not DEFAULT_STOCK (avoid double counting)
        if stock_id != "DEFAULT_STOCK":
            self.committed_shares += quantity

        # Decrement the correct stock's position
        # Only reduce by what we actually have of this stock (not borrowed amount)
        owned_shares = min(quantity, current_shares)
        if owned_shares > 0:
            self.positions[stock_id] = current_shares - owned_shares

        # Log final state
        LoggingService.log_agent_state(
            agent_id=self.agent_id,
            operation="after share commit",
            agent_state=self._get_state_dict()
        )

        # Check invariants after borrowing
        if self.allow_short_selling and shares_to_borrow > 0:
            self._check_borrowed_positions_invariants()

    def _release_shares(self, quantity: float, return_borrowed: bool = True, stock_id: str = "DEFAULT_STOCK"):
        """Release committed shares after order completion.

        Args:
            quantity: Amount of shares to release from commitment
            return_borrowed: When ``True`` the released shares reduce any
                outstanding borrow. When ``False`` (used when a sell order is
                filled), borrowed shares remain outstanding to represent the
                short position.
            stock_id: Which stock's shares to release (for multi-stock mode)
        """
        LoggingService.log_agent_state(
            agent_id=self.agent_id,
            operation=f"releasing shares ({stock_id})",
            amount=quantity,
            agent_state=self._get_state_dict(),
            outstanding_orders=self.outstanding_orders
        )

        FLOAT_TOLERANCE = 1e-10  # tolerance for floating point comparisons

        if quantity - self.committed_shares > FLOAT_TOLERANCE:
            raise ValueError(f"Cannot release more than committed: {quantity} > {self.committed_shares}")

        # Release committed shares per stock
        current_committed = self.committed_positions.get(stock_id, 0)
        self.committed_positions[stock_id] = max(0, current_committed - quantity)
        # Also update global if not DEFAULT_STOCK (avoid double counting)
        if stock_id != "DEFAULT_STOCK":
            self.committed_shares = max(0, self.committed_shares - quantity)

        if return_borrowed:
            # If we've borrowed shares for this stock, return those first
            current_borrowed = self.borrowed_positions.get(stock_id, 0)
            if current_borrowed > 0:
                shares_to_return = min(quantity, current_borrowed)
                self.borrowed_positions[stock_id] = current_borrowed - shares_to_return
                # Also update global if not DEFAULT_STOCK (avoid double counting)
                if stock_id != "DEFAULT_STOCK":
                    self.borrowed_shares -= shares_to_return
                # Only add the remaining shares (if any) to available balance for this stock
                shares_to_restore = max(0, quantity - shares_to_return)
                if shares_to_restore > 0:
                    current_position = self.positions.get(stock_id, 0)
                    self.positions[stock_id] = current_position + shares_to_restore
            else:
                # No borrowed shares, so just add everything back to this stock
                current_position = self.positions.get(stock_id, 0)
                self.positions[stock_id] = current_position + quantity

        LoggingService.log_agent_state(
            agent_id=self.agent_id,
            operation="after share release",
            agent_state=self._get_state_dict()
        )

        # Check invariants after releasing/returning borrowed shares
        if self.allow_short_selling and return_borrowed:
            self._check_borrowed_positions_invariants()

    @property
    def available_cash(self):
        """Get cash available for new orders"""
        return self.cash  # This is already the available amount, as committed are tracked separately
    
    @property
    def available_shares(self):
        """Get shares available for new orders"""
        return self.shares  # This is already the available amount, as committed are tracked separately
    
    @property
    def total_shares(self):
        """Get total shares position (available + committed) across all stocks"""
        total = 0
        for stock_id in self.positions.keys():
            total += self.positions[stock_id] + self.committed_positions.get(stock_id, 0)
        return total
    
    def update_wealth(self, prices):
        """Update agent's total wealth based on current prices, accounting for borrowed shares

        Args:
            prices: Either a single float (for backwards compatibility with single-stock)
                   or a Dict[stock_id, price] for multi-stock scenarios
        """
        # Handle both single price and multi-stock prices dict
        if isinstance(prices, dict):
            # Multi-stock: Calculate wealth across all positions
            self.last_prices = prices  # Store for margin calls

            # Skip DEFAULT_STOCK as it's an accumulator, not a real stock
            share_value = sum(
                (self.positions.get(stock_id, 0) + self.committed_positions.get(stock_id, 0) -
                 self.borrowed_positions.get(stock_id, 0)) * price
                for stock_id, price in prices.items()
                if stock_id != "DEFAULT_STOCK"
            )
            self.wealth = self.total_cash + share_value

            # For backwards compatibility, set last_price to first stock's price
            self.last_price = list(prices.values())[0] if prices else 0.0

            # Automatically handle margin requirements after price update
            self.handle_multi_stock_margin_call(prices, self.last_update_round)
        else:
            # Single-stock: Original behavior (backwards compatible)
            current_price = prices
            self.last_price = current_price

            # Net shares position (can be negative with short selling)
            net_shares = self.total_shares - self.borrowed_shares
            self.wealth = self.total_cash + (net_shares * current_price)

            # Automatically handle margin requirements after price update
            self.handle_margin_call(current_price, self.last_update_round)

    def handle_margin_call(self, current_price: float, round_number: int):
        """Force buy-to-cover when margin requirements are violated."""
        if self.borrowed_shares <= 0:
            return

        max_borrowable = self.get_max_borrowable_shares(current_price)
        if self.borrowed_shares > max_borrowable:
            excess = self.borrowed_shares - max_borrowable
            original_borrowed = self.borrowed_shares
            cost = excess * current_price

            # Execute forced buy-to-cover
            self.borrowed_shares -= excess
            self.shares += excess
            self.cash -= cost
            self.record_payment('main', -cost, 'trade', round_number)

            LoggingService.log_margin_call(
                round_number=round_number,
                agent_id=self.agent_id,
                agent_type=self.agent_type.name,
                borrowed_shares=original_borrowed,
                max_borrowable=max_borrowable,
                action="BUY_TO_COVER",
                excess_shares=excess,
                price=current_price
            )

            LoggingService.log_agent_state(
                agent_id=self.agent_id,
                operation="MARGIN CALL - FORCED BUY TO COVER",
                amount=excess,
                agent_state=self._get_state_dict(),
                outstanding_orders=self.outstanding_orders,
                order_history=self.order_history,
                is_error=True
            )

    def sync_orders(self, orders):
        """Sync agent's orders with the order repository"""
        # Track all orders by state
        self.orders = {
            # Pre-book states
            'input': {'buy': [], 'sell': []},
            'validated': {'buy': [], 'sell': []},
            'matching': {'buy': [], 'sell': []},
            'limit_matching': {'buy': [], 'sell': []},
            
            # Active states
            'pending': {'buy': [], 'sell': []},
            'active': {'buy': [], 'sell': []},
            'partially_filled': {'buy': [], 'sell': []},
            
            # Terminal states
            'filled': {'buy': [], 'sell': []},
            'cancelled': {'buy': [], 'sell': []}
        }
        
        # Categorize orders by state and side
        for order in orders:
            state = order.state.name.lower()
            if state in self.orders:
                self.orders[state][order.side].append(order)
        
        # Keep outstanding_orders for backward compatibility and quick access
        # Only include orders that are still in play
        self.outstanding_orders = {
            'buy': (
                self.orders['pending']['buy'] + 
                self.orders['active']['buy'] + 
                self.orders['partially_filled']['buy']
            ),
            'sell': (
                self.orders['pending']['sell'] + 
                self.orders['active']['sell'] + 
                self.orders['partially_filled']['sell']
            )
        }

    def record_trade(self, trade: Trade):
        """Record a trade and update statistics"""
        self.trade_history.append(trade)
        
        # Update trade statistics
        if trade.buyer_id == self.agent_id:
            self.trade_stats['buys'] += 1
            self.trade_stats['buy_volume'] += trade.quantity
            self.trade_stats['buy_value'] += trade.value
            self.trade_stats['avg_buy_price'] = (
                self.trade_stats['buy_value'] / self.trade_stats['buy_volume']
                if self.trade_stats['buy_volume'] > 0 else 0.0
            )
        else:  # seller
            self.trade_stats['sells'] += 1
            self.trade_stats['sell_volume'] += trade.quantity
            self.trade_stats['sell_value'] += trade.value
            self.trade_stats['avg_sell_price'] = (
                self.trade_stats['sell_value'] / self.trade_stats['sell_volume']
                if self.trade_stats['sell_volume'] > 0 else 0.0
            )

    def get_trade_summary(self) -> dict:
        """Get summary of trading activity"""
        summary = {
            'total_trades': len(self.trade_history),
            'net_volume': self.trade_stats['buy_volume'] - self.trade_stats['sell_volume'],
            'net_value': self.trade_stats['buy_value'] - self.trade_stats['sell_value'],
            'avg_trade_size': (
                (self.trade_stats['buy_volume'] + self.trade_stats['sell_volume']) / 
                len(self.trade_history) if self.trade_history else 0
            ),
            'recent_trades': self.trade_history[-5:],  # Show last 5 trades
            **self.trade_stats
        }
        
        # Add realized P&L
        if self.trade_stats['sell_volume'] > 0:
            summary['realized_pnl'] = (
                self.trade_stats['sell_value'] - 
                (self.trade_stats['sell_volume'] * 
                 (self.trade_stats['buy_value'] / self.trade_stats['buy_volume']))
                if self.trade_stats['buy_volume'] > 0 else 0.0
            )
        
        return summary

    def verify_state(self):
        """Verify agent state consistency"""
        state_valid = True
        
        # Check cash commitments
        committed_cash_from_orders = sum(
            order.current_cash_commitment or 0  # Handle None case
            for order in self.outstanding_orders['buy']
        )
        
        # Verify commitment matches
        if abs(committed_cash_from_orders - self.committed_cash) > 0.01:
            LoggingService.log_agent_state(
                agent_id=self.agent_id,
                operation="CASH COMMITMENT MISMATCH",
                amount=(f"Orders: {committed_cash_from_orders:.2f}, "
                       f"State: {self.committed_cash:.2f}"),
                agent_state=self._get_state_dict(),
                outstanding_orders=self.outstanding_orders,
                order_history=self.order_history,
                is_error=True
            )
            state_valid = False
        
        # Check for negative cash position
        if self.cash < -0.01:  # Using small tolerance for float comparison
            LoggingService.log_agent_state(
                agent_id=self.agent_id,
                operation="NEGATIVE AVAILABLE CASH NOT ALLOWED",
                amount=(f"Available cash: {self.cash:.2f}, "
                       f"Total cash: {self.total_cash:.2f} "
                       f"(including {self.committed_cash:.2f} committed, "
                       f"{self.dividend_cash:.2f} dividends)"),
                agent_state=self._get_state_dict(),
                outstanding_orders=self.outstanding_orders,
                order_history=self.order_history,
                is_error=True
            )
            state_valid = False
        
        # Check for negative share position when short selling is not allowed
        if not self.allow_short_selling and self.total_shares < 0:
            LoggingService.log_agent_state(
                agent_id=self.agent_id,
                operation="NEGATIVE TOTAL SHARE POSITION NOT ALLOWED",
                amount=(f"Total shares: {self.total_shares} "
                       f"(Available: {self.shares}, Committed: {self.committed_shares}), "
                       f"Short selling disabled"),
                agent_state=self._get_state_dict(),
                outstanding_orders=self.outstanding_orders,
                order_history=self.order_history,
                is_error=True
            )
            state_valid = False
        
        # Check for negative available shares when short selling is not allowed
        if not self.allow_short_selling and self.shares < 0:
            LoggingService.log_agent_state(
                agent_id=self.agent_id,
                operation="NEGATIVE AVAILABLE SHARES NOT ALLOWED",
                amount=(f"Available shares: {self.shares}, "
                       f"Total shares: {self.total_shares} "
                       f"(including {self.committed_shares} committed)"),
                agent_state=self._get_state_dict(),
                outstanding_orders=self.outstanding_orders,
                order_history=self.order_history,
                is_error=True
            )
            state_valid = False
        
        # Verify share position matches trade history
        if not self.verify_share_position():
            state_valid = False
        
        # Verify cash position matches payment/trade history
        if not self.verify_cash_position():
            state_valid = False
        
        # Check for borrowed shares when short selling is not allowed
        if not self.allow_short_selling and self.borrowed_shares > 0:
            LoggingService.log_agent_state(
                agent_id=self.agent_id,
                operation="BORROWED SHARES NOT ALLOWED",
                amount=f"Borrowed shares: {self.borrowed_shares}, Short selling disabled",
                agent_state=self._get_state_dict(),
                outstanding_orders=self.outstanding_orders,
                order_history=self.order_history,
                is_error=True
            )
            state_valid = False
        
        # Modify the check for negative position to account for borrowed shares
        if not self.allow_short_selling and (self.total_shares - self.borrowed_shares) < 0:
            LoggingService.log_agent_state(
                agent_id=self.agent_id,
                operation="NEGATIVE NET SHARE POSITION NOT ALLOWED",
                amount=(f"Net shares: {self.total_shares - self.borrowed_shares} "
                       f"(Total: {self.total_shares}, Borrowed: {self.borrowed_shares}), "
                       f"Short selling disabled"),
                agent_state=self._get_state_dict(),
                outstanding_orders=self.outstanding_orders,
                order_history=self.order_history,
                is_error=True
            )
            state_valid = False
        
        # Check margin requirements and trigger margin call if needed
        if self.borrowed_shares > 0 and hasattr(self, 'last_price'):
            self.handle_margin_call(self.last_price, self.last_update_round)

        # Check borrowed positions invariants
        if self.allow_short_selling:
            invariants_valid = self._check_borrowed_positions_invariants()
            state_valid = state_valid and invariants_valid

        return state_valid

    def verify_share_position(self) -> bool:
        """
        Verify that current share position matches initial position plus net trades
        Returns True if position matches trade history, False otherwise
        """
        net_trade_position = 0
        
        # Calculate net position from trades
        for trade in self.trade_history:
            if trade.buyer_id == self.agent_id:
                net_trade_position += trade.quantity
            elif trade.seller_id == self.agent_id:
                net_trade_position -= trade.quantity
            
        expected_position = self.initial_shares + net_trade_position
        
        # Check if current position matches expected position
        if abs(expected_position - self.shares) > 0.01:  # Using small tolerance for float comparison
            LoggingService.log_agent_state(
                agent_id=self.agent_id,
                operation="SHARE POSITION MISMATCH",
                amount=(f"Expected: {expected_position} "
                       f"(Initial: {self.initial_shares} + Net Trades: {net_trade_position}), "
                       f"Actual: {self.shares}"),
                agent_state=self._get_state_dict(),
                outstanding_orders=self.outstanding_orders,
                order_history=self.order_history,
                is_error=True
            )
            return False
        
        return True

    def verify_cash_position(self) -> bool:
        """
        Verify that current cash position matches initial position plus net payments and trades
        Returns True if position matches payment/trade history, False otherwise
        """
        # Start with initial positions
        expected_main_cash = self.initial_cash
        expected_dividend_cash = self.initial_dividend_cash
        
        # Add up all payments from history
        for payment_type in ['interest', 'dividend', 'trade', 'other']:
            for payment in self.payment_history[payment_type]:
                if payment.account == "main":
                    expected_main_cash += payment.amount
                elif payment.account == "dividend":
                    expected_dividend_cash += payment.amount

        # Use small tolerance for float comparison
        FLOAT_TOLERANCE = 0.01
        
        # Check if current positions match expected positions
        main_cash_matches = abs(expected_main_cash - self.cash) <= FLOAT_TOLERANCE
        dividend_cash_matches = abs(expected_dividend_cash - self.dividend_cash) <= FLOAT_TOLERANCE
        
        if not (main_cash_matches and dividend_cash_matches):
            LoggingService.log_agent_state(
                agent_id=self.agent_id,
                operation="CASH POSITION MISMATCH",
                amount=(f"Main Cash - Expected: {expected_main_cash:.2f}, Actual: {self.cash:.2f}\n"
                       f"Dividend Cash - Expected: {expected_dividend_cash:.2f}, Actual: {self.dividend_cash:.2f}"),
                agent_state=self._get_state_dict(),
                outstanding_orders=self.outstanding_orders,
                order_history=self.order_history,
                is_error=True
            )
            return False
        
        return True

    def receive_information(self, signals: Dict[InformationType, InformationSignal]):
        """Receive, store and archive information signals"""
        # Get round number from signals (handle both single-stock and multi-stock)
        if isinstance(signals, dict) and signals.get('is_multi_stock'):
            # Multi-stock: extract round from first stock's first signal
            try:
                first_stock_signals = next(iter(signals['multi_stock_signals'].values()))
                round_number = next(iter(first_stock_signals.values())).metadata.get('round', self.last_update_round + 1)
            except StopIteration:
                # Empty signals - use last round + 1
                round_number = self.last_update_round + 1
        else:
            # Single-stock: original behavior
            try:
                round_number = next(iter(signals.values())).metadata.get('round', self.last_update_round + 1)
            except StopIteration:
                # Empty signals - use last round + 1
                round_number = self.last_update_round + 1
        
        # Log received signals
        self._log_information_state(
            operation="RECEIVING SIGNALS",
            round_number=round_number,
            signals=signals
        )
        
        # Archive current signals before updating
        if self.private_signals:
            self.signal_history[self.last_update_round] = self.private_signals.copy()
            self._log_information_state(
                operation="ARCHIVING SIGNALS",
                round_number=self.last_update_round,
                signals=self.private_signals
            )
        
        # Update current signals
        self.private_signals = signals
        self.last_update_round = round_number

    def _log_information_state(self, operation: str, round_number: int, signals: Dict[InformationType, InformationSignal]):
        """Helper method to log information state changes
        
        Args:
            operation: Description of what's happening (e.g., 'RECEIVING', 'ARCHIVING')
            round_number: Current round number
            signals: Signal dictionary being logged
        """
        message = [f"\n========== Agent {self.agent_id} {operation} (Round {round_number}) =========="]

        # Handle multi-stock signal structure
        if isinstance(signals, dict) and signals.get('is_multi_stock'):
            message.append(f"\n[MULTI-STOCK MODE]")
            for stock_id, stock_signals in signals.get('multi_stock_signals', {}).items():
                message.append(f"\n\nStock: {stock_id}")
                for info_type, signal in stock_signals.items():
                    message.extend([
                        f"\n  {info_type.value}:",
                        f"\n   - Value: {signal.value}",
                        f"\n   - Reliability: {signal.reliability}",
                        f"\n   - Metadata: {signal.metadata}"
                    ])
        else:
            # Single-stock: original behavior
            for info_type, signal in signals.items():
                message.extend([
                    f"\n{info_type.value}:",
                    f"\n - Value: {signal.value}",
                    f"\n - Reliability: {signal.reliability}",
                    f"\n - Metadata: {signal.metadata}"
                ])

        full_message = ''.join(message)
        self.info_signals_logger.info(full_message)
    
    def get_signal(self, info_type: InformationType, round_number: Optional[int] = None) -> Optional[InformationSignal]:
        """Get information signal, optionally from history"""
        if round_number is None:
            return self.private_signals.get(info_type)
        return self.signal_history.get(round_number, {}).get(info_type)
    
    def get_signal_history(self, info_type: InformationType, lookback: int = None) -> Dict[int, InformationSignal]:
        """Get historical signals for a specific type
        
        Args:
            info_type: Type of information to retrieve
            lookback: Number of rounds to look back (None for all)
        """
        history = {
            round_num: signals[info_type]
            for round_num, signals in self.signal_history.items()
            if info_type in signals
        }
        
        if lookback is not None:
            rounds = sorted(history.keys(), reverse=True)[:lookback]
            history = {round_num: history[round_num] for round_num in rounds}
            
        return history

    def clear_signals(self):
        """Clear current signals but preserve history"""
        self.private_signals.clear()

    def set_info_capability(self, info_type: InformationType, capability: InfoCapability):
        """Set information capability for specific information type"""
        self.info_capabilities[info_type] = capability
    
    def get_info_capability(self, info_type: InformationType) -> Optional[InfoCapability]:
        """Get capability for specific information type"""
        return self.info_capabilities.get(info_type)
    @property
    def total_cash(self):
        """Get total cash across all accounts (available + committed + dividend)"""
        return self.cash + self.committed_cash + self.dividend_cash
    
    @property
    def total_available_cash(self):
        """Get total available cash (main account, dividend account is not for trading)"""
        return self.cash

    def record_payment(
        self,
        account: str,
        amount: float,
        payment_type: Literal['interest', 'dividend', 'trade', 'borrow_fee', 'redemption', 'other'],
        round_number: int,
        stock_id: Optional[str] = None,
    ) -> None:
        """Record a payment in the agent's history.

        Args:
            account: Account affected ("main" or "dividend").
            amount: Payment amount. Negative values represent cash outflows,
                such as dividend obligations on short positions.
            payment_type: Type of payment.
            round_number: Simulation round when the payment occurred.
            stock_id: Optional stock identifier for multi-stock scenarios.
        """
        payment = Payment(
            round_number=round_number,
            amount=amount,
            account=account,
            payment_type=payment_type,
            stock_id=stock_id,
        )

        if payment_type not in self.payment_history:
            self.payment_history['other'].append(payment)
        else:
            self.payment_history[payment_type].append(payment)

    def get_max_borrowable_shares(self, current_price: float) -> float:
        """Calculate maximum shares that can be borrowed based on margin requirements
        
        Args:
            current_price: Current market price of shares
            
        Returns:
            float: Maximum number of shares that can be borrowed
        """
        if not self.allow_short_selling:
            return 0
        
        if self.margin_base == "cash":
            # Base on available cash
            collateral = self.cash
        else:  # "wealth"
            # Base on total wealth (including existing shares)
            collateral = self.total_cash + (self.total_shares - self.total_borrowed_shares) * current_price
        
        # Calculate maximum position based on margin requirement
        # This formula ensures that: 
        # value_of_borrowed_shares ≤ collateral / margin_requirement
        max_borrowable = collateral / (current_price * self.margin_requirement)
        
        # Ensure non-negative result and respect position limit if set
        max_borrowable = max(0, max_borrowable)
        
        if self.position_limit is not None:
            # Adjust for existing position - consider both long and short
            net_position = self.total_shares - self.total_borrowed_shares
            if net_position < 0:
                # Already short, limit additional borrowing
                max_borrowable = min(max_borrowable, self.position_limit + net_position)
            else:
                # Long or neutral, can borrow up to limit
                max_borrowable = min(max_borrowable, self.position_limit)

        return max_borrowable

    def get_portfolio_margin_status(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio-wide margin status for multi-stock scenarios

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            Dict containing:
                - collateral: Total collateral value (cash or wealth-based)
                - borrowed_value: Total market value of all borrowed positions
                - net_position_value: Total value of net positions (long - short)
                - max_borrowable_value: Maximum value that can be borrowed
                - current_borrowable_shares: Dict of max shares per stock
                - margin_ratio: Current margin ratio (collateral / borrowed_value)
                - is_margin_violated: Whether margin requirements are violated
                - excess_borrowed_value: How much over limit (if violated)
        """
        if not self.allow_short_selling:
            return {
                'collateral': 0,
                'borrowed_value': 0,
                'net_position_value': 0,
                'max_borrowable_value': 0,
                'current_borrowable_shares': {},
                'margin_ratio': float('inf'),
                'is_margin_violated': False,
                'excess_borrowed_value': 0
            }

        # Calculate net position value (for reporting and collateral calculation)
        # Skip DEFAULT_STOCK as it's an accumulator, not a real stock
        net_position_value = sum(
            (self.positions.get(stock_id, 0) +
             self.committed_positions.get(stock_id, 0) -
             self.borrowed_positions.get(stock_id, 0)) * price
            for stock_id, price in prices.items()
            if stock_id != "DEFAULT_STOCK"
        )

        # Calculate collateral based on margin base setting
        if self.margin_base == "cash":
            collateral = self.cash
        else:  # "wealth"
            # Portfolio value: cash + net position value
            collateral = self.total_cash + net_position_value

        # Calculate total value of borrowed positions across all stocks
        # Skip DEFAULT_STOCK as it's an accumulator, not a real stock
        borrowed_value = sum(
            self.borrowed_positions.get(stock_id, 0) * price
            for stock_id, price in prices.items()
            if stock_id != "DEFAULT_STOCK"
        )

        # Maximum total value that can be borrowed
        max_borrowable_value = collateral / self.margin_requirement if self.margin_requirement > 0 else 0

        # Calculate max borrowable shares per stock (for reference)
        current_borrowable_shares = {}
        for stock_id, price in prices.items():
            # Skip DEFAULT_STOCK as it's an accumulator, not a real stock
            if stock_id == "DEFAULT_STOCK":
                continue
            if price > 0:
                # Allocate proportionally or use simple division
                current_borrowable_shares[stock_id] = max_borrowable_value / price

        # Check if margin is violated
        is_margin_violated = borrowed_value > max_borrowable_value
        excess_borrowed_value = max(0, borrowed_value - max_borrowable_value)

        # Calculate margin ratio (infinity if no borrowed positions)
        margin_ratio = collateral / borrowed_value if borrowed_value > 0 else float('inf')

        return {
            'collateral': collateral,
            'borrowed_value': borrowed_value,
            'net_position_value': net_position_value,
            'max_borrowable_value': max_borrowable_value,
            'current_borrowable_shares': current_borrowable_shares,
            'margin_ratio': margin_ratio,
            'is_margin_violated': is_margin_violated,
            'excess_borrowed_value': excess_borrowed_value
        }

    def handle_multi_stock_margin_call(self, prices: Dict[str, float], round_number: int):
        """Force buy-to-cover across multiple stocks when margin requirements are violated

        Args:
            prices: Dict mapping stock_id to current price
            round_number: Current round number for logging
        """
        # Check if there are any borrowed positions
        if self.total_borrowed_shares <= 0:
            return

        # Get portfolio margin status
        margin_status = self.get_portfolio_margin_status(prices)

        if not margin_status['is_margin_violated']:
            return  # No margin call needed

        # Calculate total value that needs to be covered
        excess_value = margin_status['excess_borrowed_value']

        # Strategy: Buy to cover proportionally across all borrowed positions
        # This maintains the relative composition of the short portfolio

        total_borrowed_value = margin_status['borrowed_value']
        stocks_to_cover = []

        for stock_id, price in prices.items():
            # Skip DEFAULT_STOCK as it's an accumulator, not a real stock
            if stock_id == "DEFAULT_STOCK":
                continue

            borrowed_shares = self.borrowed_positions.get(stock_id, 0)
            if borrowed_shares <= 0 or price <= 0:
                continue

            # Calculate this stock's proportion of total borrowed value
            stock_borrowed_value = borrowed_shares * price
            proportion = stock_borrowed_value / total_borrowed_value if total_borrowed_value > 0 else 0

            # Calculate shares to cover for this stock
            value_to_cover = excess_value * proportion
            shares_to_cover = value_to_cover / price if price > 0 else 0

            # Ensure we don't try to cover more than we have borrowed
            shares_to_cover = min(shares_to_cover, borrowed_shares)

            if shares_to_cover > 0:
                stocks_to_cover.append({
                    'stock_id': stock_id,
                    'shares': shares_to_cover,
                    'price': price,
                    'value': shares_to_cover * price,
                    'original_borrowed': borrowed_shares
                })

        # Execute buy-to-cover for each stock
        total_cost = 0
        for cover_info in stocks_to_cover:
            stock_id = cover_info['stock_id']
            shares = cover_info['shares']
            price = cover_info['price']
            cost = shares * price

            # Update positions
            self.borrowed_positions[stock_id] = max(0, self.borrowed_positions.get(stock_id, 0) - shares)
            # Also update DEFAULT_STOCK accumulator if not already DEFAULT_STOCK (avoid double counting)
            if stock_id != "DEFAULT_STOCK":
                self.borrowed_shares = max(0, self.borrowed_shares - shares)

            self.positions[stock_id] = self.positions.get(stock_id, 0) + shares
            self.cash -= cost
            total_cost += cost

            # Log margin call for this stock
            LoggingService.log_margin_call(
                round_number=round_number,
                agent_id=self.agent_id,
                agent_type=self.agent_type.name,
                borrowed_shares=cover_info['original_borrowed'],
                max_borrowable=margin_status['max_borrowable_value'] / price,  # Approximate
                action=f"BUY_TO_COVER_{stock_id}",
                excess_shares=shares,
                price=price
            )

        # Record the payment
        if total_cost > 0:
            self.record_payment('main', -total_cost, 'trade', round_number)

        # Log overall margin call event
        LoggingService.log_agent_state(
            agent_id=self.agent_id,
            operation="MULTI-STOCK MARGIN CALL - FORCED BUY TO COVER",
            amount=f"{len(stocks_to_cover)} stocks, total cost: {total_cost:.2f}",
            agent_state=self._get_state_dict(),
            outstanding_orders=self.outstanding_orders,
            order_history=self.order_history,
            is_error=True
        )

        # Check invariants after margin call covering
        self._check_borrowed_positions_invariants()
