"""Calculator for aggregate short interest across all agents."""


def calculate_short_interest(agent_repository, current_price: float) -> float:
    """
    Calculate aggregate short interest across all agents.

    Short interest represents the total number of borrowed shares across
    all market participants. This is a key market metric for understanding
    the level of short selling activity.

    Args:
        agent_repository: Repository containing all agents with their positions
        current_price: Current market price for position valuation

    Returns:
        Total borrowed shares across all agents

    Example:
        >>> short_interest = calculate_short_interest(agent_repo, 28.0)
        >>> print(f"Total borrowed shares: {short_interest}")
    """
    return sum(
        agent_repository.get_agent_state_snapshot(
            agent_id,
            current_price
        ).borrowed_shares
        for agent_id in agent_repository.get_all_agent_ids()
    )
