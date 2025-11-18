"""Agent state verification module.

This module handles all verification and invariant checking for agent state,
extracted from BaseAgent as part of issue #57 refactoring.

Pattern: Follows the SimulationVerifier pattern from issue #25
"""

from typing import Dict, TYPE_CHECKING
import warnings
from services.logging_service import LoggingService
from constants import FLOAT_TOLERANCE, CASH_MATCHING_TOLERANCE

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


class AgentVerifier:
    """Handles verification of agent state consistency and invariants."""

    def __init__(self, agent: 'BaseAgent'):
        """Initialize verifier with agent reference.

        Args:
            agent: The BaseAgent instance to verify
        """
        self.agent = agent

    def verify_state(self) -> bool:
        """Verify agent state consistency.

        Checks all invariants and constraints on the agent's state including:
        - Cash commitment consistency
        - Negative cash/share positions
        - Share position vs trade history
        - Cash position vs payment history
        - Borrowed positions (if short selling enabled)
        - Margin requirements

        Returns:
            bool: True if all checks pass, False otherwise
        """
        state_valid = True

        # Check cash commitments
        committed_cash_from_orders = sum(
            order.current_cash_commitment or 0  # Handle None case
            for order in self.agent.outstanding_orders['buy']
        )

        # Verify commitment matches
        if abs(committed_cash_from_orders - self.agent.committed_cash) > 0.01:
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="CASH COMMITMENT MISMATCH",
                amount=(f"Orders: {committed_cash_from_orders:.2f}, "
                       f"State: {self.agent.committed_cash:.2f}"),
                agent_state=self.agent._get_state_dict(),
                outstanding_orders=self.agent.outstanding_orders,
                order_history=self.agent.order_history,
                is_error=True
            )
            state_valid = False

        # Check for negative cash position
        if self.agent.cash < -0.01:  # Using small tolerance for float comparison
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="NEGATIVE AVAILABLE CASH NOT ALLOWED",
                amount=(f"Available cash: {self.agent.cash:.2f}, "
                       f"Total cash: {self.agent.total_cash:.2f} "
                       f"(including {self.agent.committed_cash:.2f} committed, "
                       f"{self.agent.dividend_cash:.2f} dividends)"),
                agent_state=self.agent._get_state_dict(),
                outstanding_orders=self.agent.outstanding_orders,
                order_history=self.agent.order_history,
                is_error=True
            )
            state_valid = False

        # Check for negative share position when short selling is not allowed
        if not self.agent.allow_short_selling and self.agent.total_shares < 0:
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="NEGATIVE TOTAL SHARE POSITION NOT ALLOWED",
                amount=(f"Total shares: {self.agent.total_shares} "
                       f"(Available: {self.agent.shares}, Committed: {self.agent.committed_shares}), "
                       f"Short selling disabled"),
                agent_state=self.agent._get_state_dict(),
                outstanding_orders=self.agent.outstanding_orders,
                order_history=self.agent.order_history,
                is_error=True
            )
            state_valid = False

        # Check for negative available shares when short selling is not allowed
        if not self.agent.allow_short_selling and self.agent.shares < 0:
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="NEGATIVE AVAILABLE SHARES NOT ALLOWED",
                amount=(f"Available shares: {self.agent.shares}, "
                       f"Total shares: {self.agent.total_shares} "
                       f"(including {self.agent.committed_shares} committed)"),
                agent_state=self.agent._get_state_dict(),
                outstanding_orders=self.agent.outstanding_orders,
                order_history=self.agent.order_history,
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
        if not self.agent.allow_short_selling and self.agent.borrowed_shares > 0:
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="BORROWED SHARES NOT ALLOWED",
                amount=f"Borrowed shares: {self.agent.borrowed_shares}, Short selling disabled",
                agent_state=self.agent._get_state_dict(),
                outstanding_orders=self.agent.outstanding_orders,
                order_history=self.agent.order_history,
                is_error=True
            )
            state_valid = False

        # Modify the check for negative position to account for borrowed shares
        if not self.agent.allow_short_selling and (self.agent.total_shares - self.agent.borrowed_shares) < 0:
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="NEGATIVE NET SHARE POSITION NOT ALLOWED",
                amount=(f"Net shares: {self.agent.total_shares - self.agent.borrowed_shares} "
                       f"(Total: {self.agent.total_shares}, Borrowed: {self.agent.borrowed_shares}), "
                       f"Short selling disabled"),
                agent_state=self.agent._get_state_dict(),
                outstanding_orders=self.agent.outstanding_orders,
                order_history=self.agent.order_history,
                is_error=True
            )
            state_valid = False

        # Check margin requirements and trigger margin call if needed
        if self.agent.borrowed_shares > 0 and hasattr(self.agent, 'last_price'):
            self.agent.handle_margin_call(self.agent.last_price, self.agent.last_update_round)

        # Check borrowed positions invariants
        if self.agent.allow_short_selling:
            invariants_valid = self.check_borrowed_positions_invariants()
            state_valid = state_valid and invariants_valid

        return state_valid

    def verify_share_position(self) -> bool:
        """Verify that current share position matches initial position plus net trades.

        Returns:
            bool: True if position matches trade history, False otherwise
        """
        net_trade_position = 0

        # Calculate net position from trades
        for trade in self.agent.trade_history:
            if trade.buyer_id == self.agent.agent_id:
                net_trade_position += trade.quantity
            elif trade.seller_id == self.agent.agent_id:
                net_trade_position -= trade.quantity

        expected_position = self.agent.initial_shares + net_trade_position

        # Check if current position matches expected position
        if abs(expected_position - self.agent.shares) > 0.01:  # Using small tolerance for float comparison
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="SHARE POSITION MISMATCH",
                amount=(f"Expected: {expected_position} "
                       f"(Initial: {self.agent.initial_shares} + Net Trades: {net_trade_position}), "
                       f"Actual: {self.agent.shares}"),
                agent_state=self.agent._get_state_dict(),
                outstanding_orders=self.agent.outstanding_orders,
                order_history=self.agent.order_history,
                is_error=True
            )
            return False

        return True

    def verify_cash_position(self) -> bool:
        """Verify that current cash position matches initial position plus net payments and trades.

        Returns:
            bool: True if position matches payment/trade history, False otherwise
        """
        # Start with initial positions
        expected_main_cash = self.agent.initial_cash
        expected_dividend_cash = self.agent.initial_dividend_cash

        # Add up all payments from history
        for payment_type in ['interest', 'dividend', 'trade', 'other']:
            for payment in self.agent.payment_history[payment_type]:
                if payment.account == "main":
                    expected_main_cash += payment.amount
                elif payment.account == "dividend":
                    expected_dividend_cash += payment.amount

        # Account for borrowed cash (leverage)
        # Borrowed cash increases available cash but is a liability, not income
        # So it's not in payment history but affects self.cash
        expected_main_cash += self.agent.borrowed_cash

        # Check if current positions match expected positions (using 1 cent tolerance for cash aggregates)
        main_cash_matches = abs(expected_main_cash - self.agent.cash) <= CASH_MATCHING_TOLERANCE
        dividend_cash_matches = abs(expected_dividend_cash - self.agent.dividend_cash) <= CASH_MATCHING_TOLERANCE

        if not (main_cash_matches and dividend_cash_matches):
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="CASH POSITION MISMATCH",
                amount=(f"Main Cash - Expected: {expected_main_cash:.2f} (including ${self.agent.borrowed_cash:.2f} borrowed), "
                       f"Actual: {self.agent.cash:.2f}\n"
                       f"Dividend Cash - Expected: {expected_dividend_cash:.2f}, Actual: {self.agent.dividend_cash:.2f}"),
                agent_state=self.agent._get_state_dict(),
                outstanding_orders=self.agent.outstanding_orders,
                order_history=self.agent.order_history,
                is_error=True
            )
            return False

        return True

    def check_borrowed_positions_invariants(self) -> bool:
        """Verify borrowed positions invariants are maintained.

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
        all_positions = self.agent.borrowed_positions

        # Invariant 1: No negative borrowed positions
        for stock_id, shares in all_positions.items():
            if shares < 0:
                error_msg = f"INVARIANT VIOLATION: Negative borrowed position for {stock_id}: {shares}"
                LoggingService.log_agent_state(
                    agent_id=self.agent.agent_id,
                    operation="INVARIANT_VIOLATION",
                    amount=error_msg,
                    agent_state=self.agent._get_state_dict(),
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
                    agent_id=self.agent.agent_id,
                    operation="INVARIANT_VIOLATION",
                    amount=error_msg,
                    agent_state=self.agent._get_state_dict(),
                    is_error=True
                )
                # Warning instead of assertion failure in production
                # This allows simulation to continue while logging the issue
                warnings.warn(error_msg, RuntimeWarning)
                return False

        # Invariant 3: total_borrowed_shares consistency
        expected_total = self.agent.total_borrowed_shares
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
                agent_id=self.agent.agent_id,
                operation="INVARIANT_VIOLATION",
                amount=error_msg,
                agent_state=self.agent._get_state_dict(),
                is_error=True
            )
            warnings.warn(error_msg, RuntimeWarning)
            return False

        return True

    def check_leverage_invariants(self, prices: Dict[str, float] = None) -> bool:
        """Verify leverage invariants are maintained.

        This defensive check ensures internal consistency of the leverage trading system,
        particularly cash borrowing and margin calculations.

        Invariants checked:
        1. borrowed_cash is never negative
        2. borrowed_cash matches CashLendingRepository records (if repo exists)
        3. If borrowed_cash > 0, agent must have leverage_ratio > 1.0
        4. If borrowed_cash > 0, cash_lending_repo must be set
        5. leverage_interest_paid is never negative
        6. If prices provided, equity calculation is consistent

        Args:
            prices: Optional price dict for equity consistency checks

        Returns:
            bool: True if all invariants pass

        Raises:
            AssertionError: If any critical invariant is violated
        """
        # Invariant 1: No negative borrowed cash
        if self.agent.borrowed_cash < -1e-10:  # Small tolerance for floating point
            error_msg = f"INVARIANT VIOLATION: Negative borrowed cash: ${self.agent.borrowed_cash:.2f}"
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="LEVERAGE_INVARIANT_VIOLATION",
                amount=error_msg,
                agent_state=self.agent._get_state_dict(),
                is_error=True
            )
            assert False, error_msg

        # Invariant 2: Repository consistency
        if self.agent.cash_lending_repo and self.agent.borrowed_cash > 0:
            repo_borrowed = self.agent.cash_lending_repo.get_borrowed(self.agent.agent_id)
            tolerance = 1e-6  # Very small tolerance for floating point
            if abs(self.agent.borrowed_cash - repo_borrowed) > tolerance:
                error_msg = (
                    f"INVARIANT VIOLATION: Borrowed cash mismatch. "
                    f"Agent tracking: ${self.agent.borrowed_cash:.2f}, "
                    f"Repository tracking: ${repo_borrowed:.2f}"
                )
                LoggingService.log_agent_state(
                    agent_id=self.agent.agent_id,
                    operation="LEVERAGE_INVARIANT_VIOLATION",
                    amount=error_msg,
                    agent_state=self.agent._get_state_dict(),
                    is_error=True
                )
                warnings.warn(error_msg, RuntimeWarning)
                return False

        # Invariant 3: Borrowed cash requires leverage enabled
        if self.agent.borrowed_cash > 1e-6 and self.agent.leverage_ratio <= 1.0:
            error_msg = (
                f"INVARIANT VIOLATION: Agent has borrowed cash (${self.agent.borrowed_cash:.2f}) "
                f"but leverage_ratio is {self.agent.leverage_ratio:.2f} (should be > 1.0)"
            )
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="LEVERAGE_INVARIANT_VIOLATION",
                amount=error_msg,
                agent_state=self.agent._get_state_dict(),
                is_error=True
            )
            warnings.warn(error_msg, RuntimeWarning)
            return False

        # Invariant 4: Borrowed cash requires repository
        if self.agent.borrowed_cash > 1e-6 and self.agent.cash_lending_repo is None:
            error_msg = (
                f"INVARIANT VIOLATION: Agent has borrowed cash (${self.agent.borrowed_cash:.2f}) "
                f"but cash_lending_repo is None"
            )
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="LEVERAGE_INVARIANT_VIOLATION",
                amount=error_msg,
                agent_state=self.agent._get_state_dict(),
                is_error=True
            )
            warnings.warn(error_msg, RuntimeWarning)
            return False

        # Invariant 5: No negative interest paid
        if self.agent.leverage_interest_paid < -1e-10:
            error_msg = f"INVARIANT VIOLATION: Negative interest paid: ${self.agent.leverage_interest_paid:.2f}"
            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="LEVERAGE_INVARIANT_VIOLATION",
                amount=error_msg,
                agent_state=self.agent._get_state_dict(),
                is_error=True
            )
            warnings.warn(error_msg, RuntimeWarning)
            return False

        # Invariant 6: Equity consistency (if prices provided)
        if prices and self.agent.borrowed_cash > 0:
            try:
                # Equity should equal wealth (which already subtracts borrowed_cash)
                calculated_equity = self.agent.get_equity(prices)
                # Wealth should be consistent
                share_value = sum(
                    (self.agent.positions.get(stock_id, 0) + self.agent.committed_positions.get(stock_id, 0) -
                     self.agent.borrowed_positions.get(stock_id, 0)) * price
                    for stock_id, price in prices.items()
                    if stock_id != "DEFAULT_STOCK"
                )
                expected_wealth = self.agent.total_cash + share_value - self.agent.borrowed_cash

                tolerance = 0.01
                if abs(calculated_equity - expected_wealth) > tolerance:
                    error_msg = (
                        f"INVARIANT VIOLATION: Equity calculation inconsistency. "
                        f"Calculated equity: ${calculated_equity:.2f}, "
                        f"Expected (from components): ${expected_wealth:.2f}"
                    )
                    LoggingService.log_agent_state(
                        agent_id=self.agent.agent_id,
                        operation="LEVERAGE_INVARIANT_VIOLATION",
                        amount=error_msg,
                        agent_state=self.agent._get_state_dict(),
                        is_error=True
                    )
                    warnings.warn(error_msg, RuntimeWarning)
                    return False
            except Exception as e:
                # Don't fail if calculation has issues, just log it
                warnings.warn(f"Could not verify equity invariant: {e}", RuntimeWarning)

        return True
