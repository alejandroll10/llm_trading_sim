from typing import Dict, Optional
from services.logging_service import LoggingService


class CashLendingRepository:
    """Repository managing a pool of lendable cash for leveraged trading.

    This mirrors the BorrowingRepository pattern but manages cash instead of shares.
    Agents can borrow cash to take leveraged long positions, paying interest on borrowed amounts.
    """

    def __init__(
        self,
        total_lendable_cash: float = float('inf'),
        allow_partial_borrows: bool = False,
        logger=None
    ) -> None:
        """Initialize the cash lending repository.

        Args:
            total_lendable_cash: Total cash available to lend (default: unlimited)
            allow_partial_borrows: If True, allow partial fills when pool has insufficient cash
            logger: Optional logger instance
        """
        self.total_lendable_cash = total_lendable_cash
        self.available_cash = total_lendable_cash
        self.borrowed: Dict[str, float] = {}
        self.allow_partial_borrows = allow_partial_borrows
        self.logger = logger or LoggingService.get_logger('cash_lending')

        # Handle case where LoggingService returns None
        if self.logger is None:
            import logging
            self.logger = logging.getLogger('cash_lending')
            self.logger.setLevel(logging.INFO)

        pool_description = "unlimited" if total_lendable_cash == float('inf') else f"${total_lendable_cash:,.2f}"
        self.logger.info(
            f"Initialized cash lending repository with {pool_description} lendable cash "
            f"(partial borrows: {'enabled' if allow_partial_borrows else 'disabled'})"
        )

    def allocate_cash(
        self,
        agent_id: str,
        amount: float,
        allow_partial: Optional[bool] = None
    ) -> float:
        """Allocate cash to an agent if available.

        Args:
            agent_id: ID of the agent requesting cash
            amount: Amount of cash requested
            allow_partial: If True, allocate up to available cash; if False, all-or-nothing.
                          If None, uses the repository's default setting.

        Returns:
            Amount of cash actually allocated (0 if none available or partial not allowed)
        """
        if amount <= 0:
            return 0.0

        # Use instance setting if not explicitly provided
        if allow_partial is None:
            allow_partial = self.allow_partial_borrows

        # Determine how much cash can be allocated
        if amount > self.available_cash:
            if not allow_partial:
                self.logger.warning(
                    f"Borrow request denied for agent {agent_id}: requested ${amount:,.2f}, "
                    f"available ${self.available_cash:,.2f} (partial fills disabled)"
                )
                return 0.0
            # Partial fill: allocate what's available
            allocated = self.available_cash
        else:
            # Full allocation
            allocated = amount

        if allocated == 0:
            self.logger.warning(
                f"Borrow request denied for agent {agent_id}: no cash available"
            )
            return 0.0

        # Update state
        self.available_cash -= allocated
        self.borrowed[agent_id] = self.borrowed.get(agent_id, 0.0) + allocated

        if allocated < amount:
            self.logger.info(
                f"Partial allocation for agent {agent_id}: requested ${amount:,.2f}, "
                f"allocated ${allocated:,.2f}. Remaining lendable cash: ${self.available_cash:,.2f}"
            )
        else:
            self.logger.info(
                f"Allocated ${allocated:,.2f} to agent {agent_id}. "
                f"Remaining lendable cash: ${self.available_cash:,.2f}"
            )

        return allocated

    def release_cash(self, agent_id: str, amount: float) -> None:
        """Return borrowed cash from an agent back to the pool.

        Args:
            agent_id: ID of the agent returning cash
            amount: Amount of cash to return

        Raises:
            ValueError: If trying to return more than borrowed
        """
        borrowed = self.borrowed.get(agent_id, 0.0)
        if amount > borrowed + 1e-10:  # Small tolerance for floating point errors
            raise ValueError(
                f"Cannot release more cash than borrowed: attempting to release ${amount:,.2f} "
                f"with only ${borrowed:,.2f} borrowed"
            )

        borrowed_after = borrowed - amount
        if borrowed_after > 1e-10:  # Keep entry if still has debt
            self.borrowed[agent_id] = borrowed_after
        else:
            self.borrowed.pop(agent_id, None)

        self.available_cash += amount
        self.logger.info(
            f"Returned ${amount:,.2f} from agent {agent_id}. "
            f"Available lendable cash: ${self.available_cash:,.2f}"
        )

    def get_borrowed(self, agent_id: str) -> float:
        """Get amount of cash currently borrowed by an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Amount of cash borrowed by the agent
        """
        return self.borrowed.get(agent_id, 0.0)

    def get_total_borrowed(self) -> float:
        """Get total cash borrowed across all agents.

        Returns:
            Total amount of cash currently borrowed
        """
        return sum(self.borrowed.values())

    def get_utilization_rate(self) -> float:
        """Get utilization rate of the lending pool.

        Returns:
            Utilization rate (0.0 to 1.0), or 0.0 if pool is unlimited
        """
        if self.total_lendable_cash == float('inf'):
            return 0.0
        if self.total_lendable_cash == 0:
            return 0.0
        return self.get_total_borrowed() / self.total_lendable_cash
