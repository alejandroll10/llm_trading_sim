from typing import Dict
from services.logging_service import LoggingService


class BorrowingRepository:
    """Repository managing a pool of lendable shares for short selling."""

    def __init__(self, total_lendable: int = 0, logger=None) -> None:
        self.total_lendable = total_lendable
        self.available_shares = total_lendable
        self.borrowed: Dict[str, int] = {}
        self.logger = logger or LoggingService.get_logger('borrowing')
        self.logger.info(
            f"Initialized borrowing repository with {total_lendable} lendable shares"
        )

    def allocate_shares(self, agent_id: str, quantity: int) -> bool:
        """Allocate shares to an agent if available.

        Returns True if allocation succeeds, False otherwise.
        """
        if quantity > self.available_shares:
            self.logger.warning(
                f"Borrow request denied for agent {agent_id}: requested {quantity}, available {self.available_shares}"
            )
            return False

        self.available_shares -= quantity
        self.borrowed[agent_id] = self.borrowed.get(agent_id, 0) + quantity
        self.logger.info(
            f"Allocated {quantity} shares to agent {agent_id}. Remaining lendable shares: {self.available_shares}"
        )
        return True

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
