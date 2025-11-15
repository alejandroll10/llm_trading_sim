"""Calculator for agent state snapshots and commitment states.

Extracts state calculation business logic from AgentRepository.
"""

from agents.agent_manager.services.agent_data_structures import (
    AgentStateSnapshot, AgentCommitmentState
)


def calculate_state_snapshot(agent, prices) -> AgentStateSnapshot:
    """Calculate complete snapshot of agent state.

    Updates agent's wealth and returns a comprehensive state snapshot including
    positions, commitments, borrowed amounts, and trading summary.

    Args:
        agent: Agent instance to snapshot
        prices: Either a single float (single-stock) or Dict[stock_id, price] (multi-stock)

    Returns:
        AgentStateSnapshot with all agent state information

    Example:
        >>> snapshot = calculate_state_snapshot(agent, 28.0)
        >>> print(f"Agent wealth: ${snapshot.wealth:.2f}")
    """
    # Update wealth based on current prices
    agent.update_wealth(prices)

    return AgentStateSnapshot(
        agent_id=agent.agent_id,
        agent_type=agent.agent_type.type_id,
        cash=agent.cash,
        dividend_cash=agent.dividend_cash,
        shares=agent.shares,
        committed_cash=agent.committed_cash,
        committed_shares=agent.committed_shares,
        total_shares=agent.total_shares,
        borrowed_shares=agent.borrowed_shares,
        net_shares=agent.total_shares - agent.borrowed_shares,
        borrowed_cash=agent.borrowed_cash,
        leverage_interest_paid=agent.leverage_interest_paid,
        wealth=agent.wealth,
        orders_by_state=agent.orders,
        trade_summary=agent.get_trade_summary()
    )


def calculate_commitment_state(agent) -> AgentCommitmentState:
    """Calculate agent's commitment-related state.

    Extracts current commitment state showing available vs committed resources
    for order validation and management.

    Args:
        agent: Agent instance to extract state from

    Returns:
        AgentCommitmentState with available and committed resources

    Example:
        >>> state = calculate_commitment_state(agent)
        >>> print(f"Available cash: ${state.available_cash:.2f}")
    """
    return AgentCommitmentState(
        agent_id=agent.agent_id,
        available_cash=agent.available_cash,
        available_shares=agent.available_shares,
        total_cash=agent.cash,
        total_shares=agent.shares,
        committed_cash=agent.committed_cash,
        committed_shares=agent.committed_shares
    )
