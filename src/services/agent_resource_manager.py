"""Resource management for agents including borrowing, commitments, and share operations.

Extracts resource/borrowing business logic from AgentRepository.
"""

from typing import Callable, Optional, Dict
from agents.agent_manager.services.agent_data_structures import CommitmentResult
from agents.agent_manager.services.borrowing_repository import BorrowingRepository
from services.logging_service import LoggingService


def commit_shares_with_borrowing(
    agent,
    share_amount: int,
    stock_id: str,
    get_borrowing_repo: Callable[[str], BorrowingRepository],
    current_price: float,
    round_number: int
) -> CommitmentResult:
    """Commit shares for an agent, borrowing if necessary.

    If partial borrows are enabled and insufficient shares are available to borrow,
    the commitment will be reduced to the maximum fillable amount.

    Args:
        agent: Agent instance
        share_amount: Number of shares to commit
        stock_id: Which stock's shares to commit (for multi-stock mode)
        get_borrowing_repo: Function to get borrowing repository for a stock_id
        current_price: Current price for the stock
        round_number: Current round number

    Returns:
        CommitmentResult with success status and details

    Example:
        >>> result = commit_shares_with_borrowing(
        ...     agent, 1000, "AAPL", repo.get_borrowing_repo, 150.0, 5
        ... )
        >>> print(f"Committed: {result.committed_amount}")
    """
    # Save original request for tracking
    original_requested = share_amount

    # Get current position for this specific stock
    # DEFENSIVE: Check if stock_id exists in positions
    if stock_id not in agent.positions:
        # Log warning but continue (might be legitimate short sell from empty position)
        # However, this could indicate a stock_id mismatch bug
        LoggingService.get_logger('agents').warning(
            f"Agent {agent.agent_id} committing shares for stock_id='{stock_id}' "
            f"which is not in positions {list(agent.positions.keys())}. "
            f"Treating as short sell from zero position. "
            f"If this is unexpected, it may indicate a stock_id mismatch bug."
        )
        current_shares = 0
    else:
        current_shares = agent.positions[stock_id]

    # Determine if we need to borrow shares
    shares_needed = max(0, share_amount - current_shares)
    allocated_shares = 0

    if shares_needed > 0:
        # Get the correct borrowing repository for this stock
        borrowing_repo = get_borrowing_repo(stock_id)

        # DEBUG: Log pool state before allocation
        LoggingService.get_logger('borrowing').warning(
            f"[BORROW_FLOW] Agent {agent.agent_id} requesting {shares_needed} shares of {stock_id}. "
            f"Pool state: available={borrowing_repo.available_shares}, "
            f"already borrowed by agent={borrowing_repo.borrowed.get(agent.agent_id, 0)}"
        )

        # Try to allocate shares (respects allow_partial_borrows setting)
        allocated_shares = borrowing_repo.allocate_shares(agent.agent_id, shares_needed)

        # DEBUG: Log allocation result
        LoggingService.get_logger('borrowing').warning(
            f"[BORROW_FLOW] Pool allocated {allocated_shares} shares to agent {agent.agent_id}. "
            f"Pool state after: available={borrowing_repo.available_shares}, "
            f"total borrowed by agent={borrowing_repo.borrowed.get(agent.agent_id, 0)}"
        )

        if allocated_shares == 0:
            # No shares available and/or partial fills disabled
            return CommitmentResult(
                success=False,
                message=f"Insufficient lendable shares for {stock_id}: requested {shares_needed}, allocated {allocated_shares}",
                requested_amount=original_requested
            )

        # Calculate the actual fillable amount
        fillable_shares = current_shares + allocated_shares

        # Check if this is a partial fill
        is_partial = fillable_shares < share_amount

        if is_partial:
            # Adjust the commitment to what we can actually fill
            share_amount = fillable_shares
            LoggingService.get_logger('agents').info(
                f"Partial borrow fill for agent {agent.agent_id} ({stock_id}): "
                f"requested {original_requested}, "
                f"fillable {share_amount} (owned: {current_shares}, borrowed: {allocated_shares})"
            )

    try:
        # DEBUG: Log agent state before commit
        LoggingService.get_logger('borrowing').warning(
            f"[BORROW_FLOW] Agent {agent.agent_id} before commit_shares: "
            f"borrowed_positions={agent.borrowed_positions.get(stock_id, 0)}, "
            f"committed_positions={agent.committed_positions.get(stock_id, 0)}, "
            f"positions={agent.positions.get(stock_id, 0)}"
        )

        agent.commit_shares(share_amount, round_number=round_number, current_price=current_price, stock_id=stock_id)

        # DEBUG: Log agent state after commit
        LoggingService.get_logger('borrowing').warning(
            f"[BORROW_FLOW] Agent {agent.agent_id} after commit_shares: "
            f"borrowed_positions={agent.borrowed_positions.get(stock_id, 0)}, "
            f"committed_positions={agent.committed_positions.get(stock_id, 0)}, "
            f"positions={agent.positions.get(stock_id, 0)}"
        )

        # Determine if this was a partial fill
        partial_fill = shares_needed > 0 and allocated_shares < shares_needed

        return CommitmentResult(
            success=True,
            message="Shares committed successfully" + (" (partial fill)" if partial_fill else ""),
            committed_amount=share_amount,
            partial_fill=partial_fill,
            requested_amount=original_requested
        )
    except ValueError as e:
        # Roll back any allocated shares on failure
        LoggingService.get_logger('borrowing').error(
            f"[BORROW_FLOW] commit_shares FAILED for agent {agent.agent_id}: {e}. "
            f"Rolling back {allocated_shares} allocated shares."
        )
        if allocated_shares > 0:
            borrowing_repo = get_borrowing_repo(stock_id)
            LoggingService.get_logger('borrowing').warning(
                f"[BORROW_FLOW] Pool before rollback: "
                f"available={borrowing_repo.available_shares}, "
                f"borrowed by agent={borrowing_repo.borrowed.get(agent.agent_id, 0)}"
            )
            borrowing_repo.release_shares(agent.agent_id, allocated_shares)
            LoggingService.get_logger('borrowing').warning(
                f"[BORROW_FLOW] Pool after rollback: "
                f"available={borrowing_repo.available_shares}, "
                f"borrowed by agent={borrowing_repo.borrowed.get(agent.agent_id, 0)}"
            )
        return CommitmentResult(False, str(e), requested_amount=original_requested)


def commit_agent_resources(
    agent,
    cash_amount: float,
    share_amount: int,
    stock_id: str,
    prices: Optional[Dict[str, float]],
    commit_shares_fn: Callable
) -> CommitmentResult:
    """Commit agent resources (cash or shares) with validation.

    Args:
        agent: Agent instance
        cash_amount: Amount of cash to commit
        share_amount: Number of shares to commit
        stock_id: Which stock's shares to commit (for multi-stock mode)
        prices: Current prices dict for validation (required for leverage validation)
        commit_shares_fn: Function to call for share commitment (handles borrowing)

    Returns:
        CommitmentResult with success status

    Example:
        >>> result = commit_agent_resources(
        ...     agent, cash_amount=1000.0, share_amount=0,
        ...     stock_id="AAPL", prices={"AAPL": 150.0}, commit_shares_fn=...
        ... )
    """
    try:
        if cash_amount > 0:
            # Validate cash commitment feasibility before attempting it
            if prices is not None:
                can_commit, error_msg = agent.can_commit_cash(cash_amount, prices)
                if not can_commit:
                    return CommitmentResult(False, error_msg)

            agent.commit_cash(cash_amount, prices=prices)
            return CommitmentResult(True, "Cash committed successfully", cash_amount)
        elif share_amount > 0:  # Changed from if to elif to avoid potential double commits
            return commit_shares_fn(agent.agent_id, share_amount, stock_id=stock_id)
        else:
            LoggingService.get_logger('agents').error(
                f"No amount specified for agent {agent.agent_id} with orders: {agent.get_trade_summary()}"
            )
    except ValueError as e:
        return CommitmentResult(False, str(e))


def release_agent_resources(
    agent,
    cash_amount: float,
    share_amount: int,
    return_borrowed: bool,
    stock_id: str,
    get_borrowing_repo: Callable[[str], BorrowingRepository]
) -> CommitmentResult:
    """Release agent resources (cash and/or shares) with borrowing pool management.

    Args:
        agent: Agent instance
        cash_amount: Amount of cash to release
        share_amount: Number of shares to release
        return_borrowed: Whether to return borrowed shares to the pool
        stock_id: Which stock's shares to release (for multi-stock mode)
        get_borrowing_repo: Function to get borrowing repository for a stock_id

    Returns:
        CommitmentResult with success status

    Example:
        >>> result = release_agent_resources(
        ...     agent, 0, 500, True, "AAPL", repo.get_borrowing_repo
        ... )
    """
    try:
        results = []
        if cash_amount > 0:
            agent._release_cash(cash_amount)
            results.append(f"Cash released: {cash_amount}")
        if share_amount > 0:
            # Track per-stock borrowed shares (not global property which only tracks DEFAULT_STOCK)
            borrowed_before = agent.borrowed_positions.get(stock_id, 0)
            agent._release_shares(share_amount, return_borrowed=return_borrowed, stock_id=stock_id)
            results.append(f"Shares released: {share_amount} ({stock_id})")
            borrowed_after = agent.borrowed_positions.get(stock_id, 0)
            returned = borrowed_before - borrowed_after
            if returned > 0:
                # Return shares to the correct stock's borrowing repository
                borrowing_repo = get_borrowing_repo(stock_id)
                borrowing_repo.release_shares(agent.agent_id, returned)

        if results:
            return CommitmentResult(True, "; ".join(results))
        return CommitmentResult(False, "No amount specified")
    except ValueError as e:
        return CommitmentResult(False, str(e))


def update_shares_with_covering(
    agent,
    amount: int,
    stock_id: str,
    get_borrowing_repo: Callable[[str], BorrowingRepository]
):
    """Update agent's share balance for a specific stock and handle short covering.

    When buying shares, automatically covers existing borrowed shares first before
    increasing owned position.

    Args:
        agent: Agent instance
        amount: Amount to change shares by (positive = buy, negative = sell)
        stock_id: Stock identifier
        get_borrowing_repo: Function to get borrowing repository for a stock_id

    Example:
        >>> # Agent has 100 borrowed shares, buys 150
        >>> update_shares_with_covering(agent, 150, "AAPL", repo.get_borrowing_repo)
        >>> # Result: 50 borrowed shares returned, agent now owns 50 shares
    """
    LoggingService.get_logger('agents').info(
        f"Updating share balance for agent {agent.agent_id}, amount: {amount}, stock: {stock_id}"
    )

    # Get current positions for this stock
    current_shares = agent.positions.get(stock_id, 0)
    current_borrowed = agent.borrowed_positions.get(stock_id, 0)

    if amount > 0 and current_borrowed > 0:
        # Use purchases to cover existing borrowed shares first
        cover = min(amount, current_borrowed)

        # DEBUG: Log covering transaction
        borrowing_repo = get_borrowing_repo(stock_id)
        LoggingService.get_logger('borrowing').warning(
            f"[SHORT_COVER] Agent {agent.agent_id} buying {amount} shares, covering {cover} borrowed. "
            f"Before: positions={current_shares}, borrowed={current_borrowed}. "
            f"Pool before: available={borrowing_repo.available_shares}, "
            f"borrowed by agent={borrowing_repo.borrowed.get(agent.agent_id, 0)}"
        )

        agent.borrowed_positions[stock_id] = current_borrowed - cover
        new_position = current_shares + (amount - cover)
        LoggingService.get_logger('agents').info(
            f"[SHARE_TRACE] Agent {agent.agent_id} update_shares_with_covering(BUY): "
            f"INCREASING {stock_id} from {current_shares} to {new_position} "
            f"(amount: {amount}, covered borrowed: {cover})"
        )
        agent._update_position(stock_id, new_position)
        if cover > 0:
            # Release shares to the correct stock's borrowing repository
            borrowing_repo.release_shares(agent.agent_id, cover)

            # DEBUG: Log pool state after cover
            LoggingService.get_logger('borrowing').warning(
                f"[SHORT_COVER] After returning {cover} shares to pool. "
                f"Agent: positions={new_position}, borrowed={agent.borrowed_positions[stock_id]}. "
                f"Pool after: available={borrowing_repo.available_shares}, "
                f"borrowed by agent={borrowing_repo.borrowed.get(agent.agent_id, 0)}"
            )
    else:
        new_position = current_shares + amount
        LoggingService.get_logger('agents').info(
            f"[SHARE_TRACE] Agent {agent.agent_id} update_shares_with_covering(BUY): "
            f"INCREASING {stock_id} from {current_shares} to {new_position} "
            f"(amount: {amount}, no borrowed to cover)"
        )
        agent._update_position(stock_id, new_position)

    LoggingService.get_logger('agents').info(
        f"Updated share balance for agent {agent.agent_id}, stock {stock_id}, new balance: {agent.positions[stock_id]}"
    )
    return agent


def redeem_shares_and_return_borrowed(
    agent,
    get_borrowing_repo: Callable[[str], BorrowingRepository]
):
    """Set agent's shares to zero after redemption and return all borrowed shares.

    Clears all positions for all stocks and returns borrowed shares to the
    lending pools.

    Args:
        agent: Agent instance
        get_borrowing_repo: Function to get borrowing repository for a stock_id

    Returns:
        Agent instance (for chaining)

    Example:
        >>> agent = redeem_shares_and_return_borrowed(agent, repo.get_borrowing_repo)
        >>> print(f"Positions cleared, had {previous_total} shares")
    """
    previous_total_shares = agent.total_shares

    # Clear positions for ALL stocks (multi-stock support)
    for stock_id in list(agent.positions.keys()):
        agent._update_position(stock_id, 0)
        # NOTE: Don't clear committed_positions - agents may still have active orders!
        # Commitments should only be released when orders are cancelled/filled.

        # Release borrowed shares if any
        if stock_id in agent.borrowed_positions and agent.borrowed_positions[stock_id] > 0:
            borrowed = agent.borrowed_positions[stock_id]
            agent.borrowed_positions[stock_id] = 0
            # Return shares to the correct stock's borrowing repository
            borrowing_repo = get_borrowing_repo(stock_id)
            borrowing_repo.release_shares(agent.agent_id, borrowed)

    LoggingService.get_logger('agents').info(
        f"Redeemed all shares for agent {agent.agent_id}, "
        f"previous balance: {previous_total_shares}, new balance: {agent.total_shares}"
    )
    return agent
