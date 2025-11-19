from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal, Any, Tuple
import numpy as np
from market.trade import Trade
from market.information.information_types import InformationType, InformationSignal, InfoCapability
from agents.agents_api import TradeDecision
import logging
from services.logging_service import LoggingService
from services.messaging_service import MessagingService
from constants import FLOAT_TOLERANCE, CASH_MATCHING_TOLERANCE
from agents.verification.agent_verifier import AgentVerifier
from agents.services.margin_service import MarginService

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
                 leverage_ratio: float = 1.0,  # 1.0 = no leverage, 2.0 = 2x leverage
                 initial_margin: float = 0.5,  # 50% down payment required
                 maintenance_margin: float = 0.25,  # 25% minimum margin (liquidation threshold)
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

        # Leverage fields (for long position leverage via cash borrowing)
        self.leverage_ratio = leverage_ratio  # Maximum leverage allowed (1.0 = no leverage)
        self.initial_margin = initial_margin  # Required down payment for leveraged positions
        self.maintenance_margin = maintenance_margin  # Minimum margin ratio before liquidation
        self.borrowed_cash: float = 0.0  # Cash borrowed for leverage
        self.leverage_interest_paid: float = 0.0  # Cumulative interest paid on borrowed cash
        self.cash_lending_repo = None  # Set during simulation initialization

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

        # Initialize services (issue #57 refactoring)
        self.verifier = AgentVerifier(self)  # Phase 1
        self.margin_service = MarginService(self)  # Phase 2

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
        # DEBUG
        import traceback
        if self.committed_positions.get("DEFAULT_STOCK", 0) > 0 and value == 0:
            print(f"[DEBUG SETTER] Agent {self.agent_id}: committed_positions[DEFAULT_STOCK] being set to 0!")
            traceback.print_stack(limit=10)
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

        REFACTORED: Delegates to AgentVerifier (issue #57 Phase 1)

        Returns:
            bool: True if all invariants pass
        """
        return self.verifier.check_borrowed_positions_invariants()

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

    def can_commit_cash(self, amount: float, prices: Dict[str, float]) -> Tuple[bool, str]:
        """Check if cash commitment is feasible without modifying state.

        This validation method checks whether the agent can commit the requested
        amount of cash, either from available cash or by borrowing (if leverage enabled).
        Unlike commit_cash(), this method does NOT modify any state - it only validates.

        Args:
            amount: Amount of cash to potentially commit
            prices: Dict mapping stock_id to current price (needed for borrowing power calculation)

        Returns:
            Tuple of (success: bool, error_message: str)
            If success=True, error_message is empty
            If success=False, error_message contains the reason for failure
        """
        # Case 1: Have enough cash - commitment is feasible
        if amount <= self.cash:
            return (True, "")

        # Case 2: Insufficient cash - check if we can borrow
        shortage = amount - self.cash

        # No leverage enabled or no lending repo - cannot proceed
        if self.leverage_ratio <= 1.0 or self.cash_lending_repo is None:
            return (False, f"Insufficient cash: need ${amount:.2f}, have ${self.cash:.2f}, no leverage available")

        # Calculate borrowing power using provided prices
        borrowing_power = self.get_available_borrowing_power(prices)

        # Check if we can borrow enough
        if shortage > borrowing_power:
            return (False,
                   f"Insufficient buying power: need ${amount:.2f}, "
                   f"have ${self.cash:.2f} cash + ${borrowing_power:.2f} borrowing power = "
                   f"${self.cash + borrowing_power:.2f} total")

        # All checks passed - commitment is feasible
        return (True, "")

    def commit_cash(self, amount: float, debug: bool = False, prices: Optional[Dict[str, float]] = None):
        """Commit cash for a trade, auto-borrowing if leverage enabled and needed.

        If the agent has leverage enabled and insufficient cash, this method will
        automatically borrow the required amount from the cash lending repository.

        Args:
            amount: Amount of cash to commit
            debug: If True, print detailed debug information
            prices: Current prices dict (required for leverage calculations if last_prices not set)

        Raises:
            ValueError: If insufficient cash/borrowing power
        """
        # Log initial state
        LoggingService.log_agent_state(
            agent_id=self.agent_id,
            operation="committing cash",
            amount=amount,
            agent_state=self._get_state_dict(),
            outstanding_orders=self.outstanding_orders
        )

        # NEW: Check if we need to borrow
        if amount > self.cash:
            if self.leverage_ratio <= 1.0 or self.cash_lending_repo is None:
                # No leverage or no lending repo - fail as before
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

                LoggingService.log_validation_error(
                    round_number=self.last_update_round,
                    agent_id=self.agent_id,
                    agent_type=self.agent_type.name,
                    error_type="INSUFFICIENT_CASH",
                    details=f"Required: {amount:.2f}, Available: {self.cash:.2f}",
                    attempted_action="COMMIT_CASH"
                )

                raise ValueError(f"Insufficient cash: {amount} > {self.cash}")

            # NEW: Leverage enabled - try to borrow
            shortage = amount - self.cash

            # Check if we have borrowing power
            # Use provided prices if available, otherwise fall back to last_prices
            prices_to_use = prices if prices is not None else getattr(self, 'last_prices', None)
            if prices_to_use is None:
                raise ValueError("Cannot compute borrowing power without current prices")

            borrowing_power = self.get_available_borrowing_power(prices_to_use)

            if shortage > borrowing_power:
                LoggingService.log_validation_error(
                    round_number=self.last_update_round,
                    agent_id=self.agent_id,
                    agent_type=self.agent_type.name,
                    error_type="INSUFFICIENT_BORROWING_POWER",
                    details=f"Need ${shortage:.2f}, have ${borrowing_power:.2f} borrowing power",
                    attempted_action="COMMIT_CASH_WITH_LEVERAGE"
                )
                raise ValueError(
                    f"Insufficient buying power: need ${amount:.2f}, "
                    f"have ${self.cash:.2f} cash + ${borrowing_power:.2f} borrowing power"
                )

            # Borrow the shortage
            actual_borrowed = self.cash_lending_repo.allocate_cash(self.agent_id, shortage)
            if actual_borrowed < shortage - 1e-10:  # Small tolerance for floating point
                raise ValueError(
                    f"Lending pool has insufficient cash: requested ${shortage:.2f}, got ${actual_borrowed:.2f}"
                )

            self.borrowed_cash += actual_borrowed
            self.cash += actual_borrowed

            if self.logger:
                self.logger.info(
                    f"Agent {self.agent_id} borrowed ${actual_borrowed:.2f} cash for leverage. "
                    f"Total borrowed cash: ${self.borrowed_cash:.2f}"
                )

            # Check invariants after borrowing
            self._check_leverage_invariants(prices_to_use)

        # Original logic: commit the cash
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
            'margin_base': self.margin_base,
            # Leverage fields
            'borrowed_cash': self.borrowed_cash,
            'leverage_ratio': self.leverage_ratio,
            'leverage_interest_paid': self.leverage_interest_paid,
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
                # Note: Don't update borrowed_shares property for non-DEFAULT_STOCK stocks
                # The property is for backward compatibility with single-stock mode only

        # Track committed shares per stock
        current_committed = self.committed_positions.get(stock_id, 0)
        self.committed_positions[stock_id] = current_committed + quantity
        # Note: Don't update committed_shares property for non-DEFAULT_STOCK stocks
        # The property is for backward compatibility with single-stock mode only

        # DEBUG
        if stock_id == "DEFAULT_STOCK":
            dict_id = id(self.committed_positions)
            print(f"[DEBUG commit_shares] Agent {self.agent_id}: dict_id={dict_id}, committed_positions[DEFAULT_STOCK] = {self.committed_positions['DEFAULT_STOCK']}")

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

        # Check against per-stock commitment (not global, which may not be updated for DEFAULT_STOCK)
        current_committed = self.committed_positions.get(stock_id, 0)
        if quantity - current_committed > FLOAT_TOLERANCE:
            raise ValueError(f"Cannot release more than committed for {stock_id}: {quantity} > {current_committed}")

        # Release committed shares per stock
        new_value = max(0, current_committed - quantity)
        self.committed_positions[stock_id] = new_value
        # Note: Don't update committed_shares property for non-DEFAULT_STOCK stocks
        # The property is for backward compatibility with single-stock mode only

        # DEBUG
        if stock_id == "DEFAULT_STOCK":
            print(f"[DEBUG release_shares] Agent {self.agent_id}: released {quantity}, new_value={new_value}, committed_positions[DEFAULT_STOCK] = {self.committed_positions['DEFAULT_STOCK']}")
            if new_value == 0 and current_committed > 0:
                import traceback
                print("[DEBUG] Commitment went to 0, traceback:")
                traceback.print_stack(limit=15)

        if return_borrowed:
            # If we've borrowed shares for this stock, return those first
            current_borrowed = self.borrowed_positions.get(stock_id, 0)
            if current_borrowed > 0:
                shares_to_return = min(quantity, current_borrowed)
                self.borrowed_positions[stock_id] = current_borrowed - shares_to_return
                # Note: Don't update borrowed_shares property for non-DEFAULT_STOCK stocks
                # The property is for backward compatibility with single-stock mode only
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
        """Update agent's total wealth based on current prices, accounting for borrowed shares and borrowed cash

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
            # MODIFIED: Subtract borrowed cash from wealth (liability)
            self.wealth = self.total_cash + share_value - self.borrowed_cash

            # For backwards compatibility, set last_price to first stock's price
            self.last_price = list(prices.values())[0] if prices else 0.0

            # Automatically handle margin requirements after price update
            # First handle short position margin calls
            self.handle_multi_stock_margin_call(prices, self.last_update_round)

            # NEW: Check leverage margin requirements (for long positions with borrowed cash)
            if self.borrowed_cash > 0:
                self.handle_leverage_margin_call(prices, self.last_update_round)
        else:
            # Single-stock: Original behavior (backwards compatible)
            current_price = prices
            self.last_price = current_price

            # Net shares position (can be negative with short selling)
            net_shares = self.total_shares - self.borrowed_shares
            # MODIFIED: Subtract borrowed cash
            self.wealth = self.total_cash + (net_shares * current_price) - self.borrowed_cash

            # Automatically handle margin requirements after price update
            self.handle_margin_call(current_price, self.last_update_round)

            # NEW: Check leverage for single stock
            if self.borrowed_cash > 0:
                self.handle_leverage_margin_call({"DEFAULT_STOCK": current_price}, self.last_update_round)

    def handle_margin_call(self, current_price: float, round_number: int):
        """Force buy-to-cover when margin requirements are violated.

        REFACTORED: Delegates to MarginService (issue #57 Phase 2)
        """
        self.margin_service.handle_margin_call(current_price, round_number)

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
        """Verify agent state consistency

        REFACTORED: Delegates to AgentVerifier (issue #57 Phase 1)
        """
        return self.verifier.verify_state()

    def verify_share_position(self) -> bool:
        """Verify that current share position matches initial position plus net trades

        REFACTORED: Delegates to AgentVerifier (issue #57 Phase 1)

        Returns:
            bool: True if position matches trade history, False otherwise
        """
        return self.verifier.verify_share_position()

    def verify_cash_position(self) -> bool:
        """Verify that current cash position matches initial position plus net payments and trades

        REFACTORED: Delegates to AgentVerifier (issue #57 Phase 1)

        Returns:
            bool: True if position matches payment/trade history, False otherwise
        """
        return self.verifier.verify_cash_position()

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
        """Calculate maximum shares that can be borrowed based on margin requirements.

        REFACTORED: Delegates to MarginService (issue #57 Phase 2)

        Args:
            current_price: Current market price of shares

        Returns:
            float: Maximum number of shares that can be borrowed
        """
        return self.margin_service.get_max_borrowable_shares(current_price)

    def get_portfolio_margin_status(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio-wide margin status for multi-stock scenarios.

        REFACTORED: Delegates to MarginService (issue #57 Phase 2)

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            Dict containing margin status information
        """
        return self.margin_service.get_portfolio_margin_status(prices)

    def handle_multi_stock_margin_call(self, prices: Dict[str, float], round_number: int):
        """Force buy-to-cover across multiple stocks when margin requirements are violated.

        REFACTORED: Delegates to MarginService (issue #57 Phase 2)

        Args:
            prices: Dict mapping stock_id to current price
            round_number: Current round number for logging
        """
        self.margin_service.handle_multi_stock_margin_call(prices, round_number)

    # ========== LEVERAGE HELPER METHODS ==========
    # These methods support leverage trading (borrowing cash for long positions)

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate equity (wealth) accounting for borrowed cash.

        REFACTORED: Delegates to MarginService (issue #57 Phase 2)

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            Agent's equity value
        """
        return self.margin_service.get_equity(prices)

    def get_gross_position_value(self, prices: Dict[str, float]) -> float:
        """Get total market value of all long positions (gross, not net).

        REFACTORED: Delegates to MarginService (issue #57 Phase 2)

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            Total value of long positions (including committed)
        """
        return self.margin_service.get_gross_position_value(prices)

    def get_leverage_margin_ratio(self, prices: Dict[str, float]) -> float:
        """Calculate current margin ratio for leverage (equity / gross_position_value).

        REFACTORED: Delegates to MarginService (issue #57 Phase 2)

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            Margin ratio (0.0 to infinity). Returns infinity if no positions held.
        """
        return self.margin_service.get_leverage_margin_ratio(prices)

    def get_available_borrowing_power(self, prices: Dict[str, float]) -> float:
        """Calculate additional cash that can be borrowed for long positions.

        REFACTORED: Delegates to MarginService (issue #57 Phase 2)

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            Amount of additional cash that can be borrowed
        """
        return self.margin_service.get_available_borrowing_power(prices)

    def is_under_leverage_margin(self, prices: Dict[str, float]) -> bool:
        """Check if agent is below maintenance margin for leverage.

        REFACTORED: Delegates to MarginService (issue #57 Phase 2)

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            True if margin call required, False otherwise
        """
        return self.margin_service.is_under_leverage_margin(prices)

    def _check_leverage_invariants(self, prices: Dict[str, float] = None) -> bool:
        """Verify leverage invariants are maintained.

        REFACTORED: Delegates to AgentVerifier (issue #57 Phase 1)

        Args:
            prices: Optional price dict for equity consistency checks

        Returns:
            bool: True if all invariants pass
        """
        return self.verifier.check_leverage_invariants(prices)

    def handle_leverage_margin_call(self, prices: Dict[str, float], round_number: int):
        """Force sell positions when leverage margin requirements violated.

        REFACTORED: Delegates to MarginService (issue #57 Phase 2)

        Args:
            prices: Dict mapping stock_id to current price
            round_number: Current round number for logging
        """
        self.margin_service.handle_leverage_margin_call(prices, round_number)
