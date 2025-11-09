from typing import Dict
from services.logging_service import LoggingService


class BorrowingRepository:
    """Repository managing a pool of lendable shares for short selling."""

    def __init__(self, total_lendable: int = 0, allow_partial_borrows: bool = False, logger=None) -> None:
        self.total_lendable = total_lendable
        self.available_shares = total_lendable
        self.borrowed: Dict[str, int] = {}
        self.allow_partial_borrows = allow_partial_borrows
        self.logger = logger or LoggingService.get_logger('borrowing')
        self.logger.info(
            f"Initialized borrowing repository with {total_lendable} lendable shares "
            f"(partial borrows: {'enabled' if allow_partial_borrows else 'disabled'})"
        )

    def allocate_shares(self, agent_id: str, quantity: int, allow_partial: bool = None) -> int:
        """Allocate shares to an agent if available.

        Args:
            agent_id: ID of the agent requesting shares
            quantity: Number of shares requested
            allow_partial: If True, allocate up to available shares; if False, all-or-nothing.
                          If None, uses the repository's default setting.

        Returns:
            Number of shares actually allocated (0 if none available or partial not allowed)
        """
        if quantity <= 0:
            return 0

        # Use instance setting if not explicitly provided
        if allow_partial is None:
            allow_partial = self.allow_partial_borrows

        # Determine how many shares can be allocated
        if quantity > self.available_shares:
            if not allow_partial:
                self.logger.warning(
                    f"Borrow request denied for agent {agent_id}: requested {quantity}, "
                    f"available {self.available_shares} (partial fills disabled)"
                )
                return 0
            # Partial fill: allocate what's available
            allocated = self.available_shares
        else:
            # Full allocation
            allocated = quantity

        if allocated == 0:
            self.logger.warning(
                f"Borrow request denied for agent {agent_id}: no shares available"
            )
            return 0

        # Update state
        self.available_shares -= allocated
        self.borrowed[agent_id] = self.borrowed.get(agent_id, 0) + allocated

        if allocated < quantity:
            self.logger.info(
                f"Partial allocation for agent {agent_id}: requested {quantity}, "
                f"allocated {allocated}. Remaining lendable shares: {self.available_shares}"
            )
        else:
            self.logger.info(
                f"Allocated {allocated} shares to agent {agent_id}. "
                f"Remaining lendable shares: {self.available_shares}"
            )

        return allocated

    def release_shares(self, agent_id: str, quantity: int) -> None:
        """Return borrowed shares from an agent back to the pool."""
        borrowed = self.borrowed.get(agent_id, 0)
        if quantity > borrowed:
            raise ValueError(
                f"Cannot release more shares than borrowed: attempting to release {quantity} with only {borrowed} borrowed"
            )

        borrowed_after = borrowed - quantity
        if borrowed_after:
            self.borrowed[agent_id] = borrowed_after
        else:
            self.borrowed.pop(agent_id, None)

        self.available_shares += quantity
        self.logger.info(
            f"Returned {quantity} shares from agent {agent_id}. Available lendable shares: {self.available_shares}"
        )

    def get_borrowed(self, agent_id: str) -> int:
        """Get number of shares currently borrowed by an agent."""
        return self.borrowed.get(agent_id, 0)
