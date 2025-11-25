"""Margin management service for agents.

This module handles all margin-related calculations and margin call execution,
extracted from BaseAgent as part of issue #57 refactoring Phase 2.

Pattern: Follows the AgentVerifier pattern from Phase 1
"""

from typing import Dict, TYPE_CHECKING
from services.logging_service import LoggingService

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


class MarginService:
    """Handles margin calculations and margin call execution for agents."""

    def __init__(self, agent: 'BaseAgent'):
        """Initialize service with agent reference.

        Args:
            agent: The BaseAgent instance to manage margins for
        """
        self.agent = agent

    # ========== SHORT SELLING MARGIN METHODS ==========

    def get_max_borrowable_shares(self, current_price: float) -> float:
        """Calculate maximum shares that can be borrowed based on margin requirements.

        Args:
            current_price: Current market price of shares

        Returns:
            float: Maximum number of shares that can be borrowed
        """
        if not self.agent.allow_short_selling:
            return 0

        if self.agent.margin_base == "cash":
            # Base on available cash
            collateral = self.agent.cash
        else:  # "wealth"
            # Base on total wealth (including existing shares)
            collateral = self.agent.total_cash + (self.agent.total_shares - self.agent.total_borrowed_shares) * current_price

        # Calculate maximum position based on margin requirement
        # This formula ensures that:
        # value_of_borrowed_shares ≤ collateral / margin_requirement
        max_borrowable = collateral / (current_price * self.agent.margin_requirement)

        # Ensure non-negative result and respect position limit if set
        max_borrowable = max(0, max_borrowable)

        if self.agent.position_limit is not None:
            # Adjust for existing position - consider both long and short
            net_position = self.agent.total_shares - self.agent.total_borrowed_shares
            if net_position < 0:
                # Already short, limit additional borrowing
                max_borrowable = min(max_borrowable, self.agent.position_limit + net_position)
            else:
                # Long or neutral, can borrow up to limit
                max_borrowable = min(max_borrowable, self.agent.position_limit)

        return max_borrowable

    def get_portfolio_margin_status(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio-wide margin status for multi-stock scenarios.

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
        if not self.agent.allow_short_selling:
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
            (self.agent.positions.get(stock_id, 0) +
             self.agent.committed_positions.get(stock_id, 0) -
             self.agent.borrowed_positions.get(stock_id, 0)) * price
            for stock_id, price in prices.items()
            if stock_id != "DEFAULT_STOCK"
        )

        # Calculate collateral based on margin base setting
        if self.agent.margin_base == "cash":
            collateral = self.agent.cash
        else:  # "wealth"
            # Portfolio value: cash + net position value
            collateral = self.agent.total_cash + net_position_value

        # Calculate total value of borrowed positions across all stocks
        # Skip DEFAULT_STOCK as it's an accumulator, not a real stock
        borrowed_value = sum(
            self.agent.borrowed_positions.get(stock_id, 0) * price
            for stock_id, price in prices.items()
            if stock_id != "DEFAULT_STOCK"
        )

        # Maximum total value that can be borrowed
        max_borrowable_value = collateral / self.agent.margin_requirement if self.agent.margin_requirement > 0 else 0

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

    def handle_margin_call(self, current_price: float, round_number: int):
        """Force buy-to-cover when margin requirements are violated.

        Note: This is the OLD direct manipulation system. It's disabled when the NEW order-based
        system is active (enable_intra_round_margin_checking=True).
        """
        # Skip if new order-based margin checking is enabled (handled at match engine level)
        # The new system creates real orders instead of direct manipulation
        if hasattr(self.agent, 'params') and self.agent.params.get('enable_intra_round_margin_checking', True):
            return

        if self.agent.borrowed_shares <= 0:
            return

        max_borrowable = self.get_max_borrowable_shares(current_price)
        if self.agent.borrowed_shares > max_borrowable:
            excess = self.agent.borrowed_shares - max_borrowable
            original_borrowed = self.agent.borrowed_shares
            cost = excess * current_price

            # Execute forced buy-to-cover
            self.agent.borrowed_shares -= excess
            self.agent.shares += excess
            self.agent.cash -= cost
            self.agent.record_payment('main', -cost, 'trade', round_number)

            LoggingService.log_margin_call(
                round_number=round_number,
                agent_id=self.agent.agent_id,
                agent_type=self.agent.agent_type.name,
                borrowed_shares=original_borrowed,
                max_borrowable=max_borrowable,
                action="BUY_TO_COVER",
                excess_shares=excess,
                price=current_price
            )

            LoggingService.log_agent_state(
                agent_id=self.agent.agent_id,
                operation="MARGIN CALL - FORCED BUY TO COVER",
                amount=excess,
                agent_state=self.agent._get_state_dict(),
                outstanding_orders=self.agent.outstanding_orders,
                order_history=self.agent.order_history,
                is_error=True
            )

    def handle_multi_stock_margin_call(self, prices: Dict[str, float], round_number: int):
        """Force buy-to-cover across multiple stocks when margin requirements are violated.

        Args:
            prices: Dict mapping stock_id to current price
            round_number: Current round number for logging
        """
        # Check if there are any borrowed positions
        if self.agent.total_borrowed_shares <= 0:
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

            borrowed_shares = self.agent.borrowed_positions.get(stock_id, 0)
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
            self.agent.borrowed_positions[stock_id] = max(0, self.agent.borrowed_positions.get(stock_id, 0) - shares)
            # Note: Don't update borrowed_shares property for non-DEFAULT_STOCK stocks
            # The property is for backward compatibility with single-stock mode only

            self.agent._update_position(stock_id, self.agent.positions.get(stock_id, 0) + shares)
            self.agent.cash -= cost
            total_cost += cost

            # Log margin call for this stock
            LoggingService.log_margin_call(
                round_number=round_number,
                agent_id=self.agent.agent_id,
                agent_type=self.agent.agent_type.name,
                borrowed_shares=cover_info['original_borrowed'],
                max_borrowable=margin_status['max_borrowable_value'] / price,  # Approximate
                action=f"BUY_TO_COVER_{stock_id}",
                excess_shares=shares,
                price=price
            )

        # Record the payment
        if total_cost > 0:
            self.agent.record_payment('main', -total_cost, 'trade', round_number)

        # Log overall margin call event
        LoggingService.log_agent_state(
            agent_id=self.agent.agent_id,
            operation="MULTI-STOCK MARGIN CALL - FORCED BUY TO COVER",
            amount=f"{len(stocks_to_cover)} stocks, total cost: {total_cost:.2f}",
            agent_state=self.agent._get_state_dict(),
            outstanding_orders=self.agent.outstanding_orders,
            order_history=self.agent.order_history,
            is_error=True
        )

        # Check invariants after margin call covering
        self.agent._check_borrowed_positions_invariants()

    # ========== LEVERAGE (CASH BORROWING) METHODS ==========

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate equity (wealth) accounting for borrowed cash.

        Equity = Total Cash + Net Share Value - Borrowed Cash

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            Agent's equity value
        """
        total_cash = self.agent.total_cash

        # Calculate net share value across all stocks
        share_value = sum(
            (self.agent.positions.get(stock_id, 0) + self.agent.committed_positions.get(stock_id, 0) -
             self.agent.borrowed_positions.get(stock_id, 0)) * price
            for stock_id, price in prices.items()
            if stock_id != "DEFAULT_STOCK"
        )

        return total_cash + share_value - self.agent.borrowed_cash

    def get_gross_position_value(self, prices: Dict[str, float]) -> float:
        """Get total market value of all long positions (gross, not net).

        This is the sum of all positive positions, not accounting for short positions.
        Used to calculate leverage margin ratio. Includes committed positions (shares in orders)
        to be consistent with equity calculation.

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            Total value of long positions (including committed)
        """
        return sum(
            (self.agent.positions.get(stock_id, 0) + self.agent.committed_positions.get(stock_id, 0)) * price
            for stock_id, price in prices.items()
            if stock_id != "DEFAULT_STOCK"
        )

    def get_leverage_margin_ratio(self, prices: Dict[str, float]) -> float:
        """Calculate current margin ratio for leverage (equity / gross_position_value).

        A lower ratio means more leverage is being used. When this falls below
        maintenance_margin, a margin call is triggered.

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            Margin ratio (0.0 to infinity). Returns infinity if no positions held.
        """
        position_value = self.get_gross_position_value(prices)
        if position_value == 0:
            return float('inf')
        return self.get_equity(prices) / position_value

    def get_available_borrowing_power(self, prices: Dict[str, float]) -> float:
        """Calculate additional cash that can be borrowed for long positions.

        This is based on the agent's equity and maximum allowed leverage ratio.
        Borrowing power = (Equity * Leverage_Ratio) - Current_Position_Value

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            Amount of additional cash that can be borrowed
        """
        if self.agent.leverage_ratio <= 1.0:
            return 0.0

        equity = self.get_equity(prices)
        gross_position_value = self.get_gross_position_value(prices)

        # Max position value allowed: equity * leverage_ratio
        max_position_value = equity * self.agent.leverage_ratio

        # Available borrowing = (max allowed - current position value)
        available = max(0, max_position_value - gross_position_value)
        return available

    def is_under_leverage_margin(self, prices: Dict[str, float]) -> bool:
        """Check if agent is below maintenance margin for leverage.

        Returns True if the agent's margin ratio has fallen below the maintenance
        margin threshold, triggering a margin call.

        Args:
            prices: Dict mapping stock_id to current price

        Returns:
            True if margin call required, False otherwise
        """
        if self.agent.borrowed_cash <= 0:
            return False
        margin_ratio = self.get_leverage_margin_ratio(prices)
        return margin_ratio < self.agent.maintenance_margin

    def handle_leverage_margin_call(self, prices: Dict[str, float], round_number: int):
        """Force sell positions when leverage margin requirements violated.

        This handles margin calls for LONG leverage (borrowed cash).
        When an agent's equity falls below the maintenance margin threshold,
        positions are liquidated proportionally to restore margin to the initial margin level.

        Args:
            prices: Dict mapping stock_id to current price
            round_number: Current round number for logging
        """
        # Check invariants before processing margin call
        self.agent._check_leverage_invariants(prices)

        # Check if there is borrowed cash
        if self.agent.borrowed_cash <= 0:
            return

        # Check if under-margined
        if not self.is_under_leverage_margin(prices):
            return  # No margin call needed

        # Calculate how much we need to liquidate
        equity = self.get_equity(prices)
        gross_position_value = self.get_gross_position_value(prices)

        # Edge case: If equity is negative or very small, liquidate everything
        if equity <= 0.01:  # Essentially bankrupt
            target_position_value = 0
            value_to_liquidate = gross_position_value
        else:
            # Target: restore to initial margin (more conservative than maintenance)
            target_position_value = equity / self.agent.initial_margin if self.agent.initial_margin > 0 else 0
            value_to_liquidate = gross_position_value - target_position_value

        if value_to_liquidate <= 0:
            return

        # Strategy: Liquidate proportionally across all long positions
        # This maintains the relative composition of the portfolio
        stocks_to_liquidate = []

        for stock_id, price in prices.items():
            if stock_id == "DEFAULT_STOCK":
                continue

            position_shares = self.agent.positions.get(stock_id, 0)
            if position_shares <= 0 or price <= 0:
                continue

            # Calculate this stock's proportion
            stock_value = position_shares * price
            proportion = stock_value / gross_position_value if gross_position_value > 0 else 0

            # Shares to sell
            value_to_sell = value_to_liquidate * proportion
            shares_to_sell = value_to_sell / price if price > 0 else 0
            shares_to_sell = min(shares_to_sell, position_shares)  # Can't sell more than we have

            if shares_to_sell > 0:
                stocks_to_liquidate.append({
                    'stock_id': stock_id,
                    'shares': shares_to_sell,
                    'price': price,
                    'value': shares_to_sell * price,
                    'original_position': position_shares
                })

        # Execute forced liquidation
        total_proceeds = 0
        repayment = 0

        for liquidate_info in stocks_to_liquidate:
            stock_id = liquidate_info['stock_id']
            shares = liquidate_info['shares']
            price = liquidate_info['price']
            proceeds = shares * price

            # Update positions - sell the shares
            self.agent._update_position(stock_id, max(0, self.agent.positions.get(stock_id, 0) - shares))
            self.agent.cash += proceeds
            total_proceeds += proceeds

            # Log margin call for this stock
            LoggingService.log_margin_call(
                round_number=round_number,
                agent_id=self.agent.agent_id,
                agent_type=self.agent.agent_type.name,
                borrowed_shares=liquidate_info['original_position'],  # Using for tracking
                max_borrowable=0,  # Not applicable for leverage margin calls
                action=f"FORCED_SELL_{stock_id}_LEVERAGE",
                excess_shares=shares,
                price=price
            )

        # Use proceeds to repay borrowed cash
        if total_proceeds > 0 and self.agent.borrowed_cash > 0:
            repayment = min(total_proceeds, self.agent.borrowed_cash)
            self.agent.cash -= repayment
            self.agent.borrowed_cash -= repayment
            if self.agent.cash_lending_repo:
                self.agent.cash_lending_repo.release_cash(self.agent.agent_id, repayment)

        # Record payment
        if total_proceeds > 0:
            self.agent.record_payment('main', total_proceeds, 'trade', round_number)

        # Log overall event
        LoggingService.log_agent_state(
            agent_id=self.agent.agent_id,
            operation="LEVERAGE MARGIN CALL - FORCED LIQUIDATION",
            amount=f"{len(stocks_to_liquidate)} stocks, total proceeds: ${total_proceeds:.2f}, repaid: ${repayment:.2f}",
            agent_state=self.agent._get_state_dict(),
            outstanding_orders=self.agent.outstanding_orders,
            order_history=self.agent.order_history,
            is_error=True
        )

        # Check invariants after margin call liquidation
        self.agent._check_leverage_invariants(prices)
