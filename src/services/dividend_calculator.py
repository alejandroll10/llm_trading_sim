"""Calculator for dividend payments across agents with payment history support."""

from typing import Dict, Tuple


def calculate_agent_dividends_received(
    agent,
    round_number: int,
    dividends: float,
    dividends_by_stock: Dict[str, float] = None,
    agent_net_shares: int = 0
) -> float:
    """
    Calculate dividends received by an agent in the previous round.

    Uses payment history when available for accurate tracking. Falls back to
    position-based calculation for scenarios without payment tracking.

    Args:
        agent: Agent instance with payment_history and positions attributes
        round_number: Current round (looks back to round_number - 1)
        dividends: Aggregate dividend per share (single-stock) or sum (multi-stock)
        dividends_by_stock: Per-stock dividends for multi-stock scenarios
        agent_net_shares: Agent's net shares for single-stock fallback calculation

    Returns:
        Total dividends received by agent, rounded to 2 decimal places

    Example:
        >>> divs = calculate_agent_dividends_received(agent, 5, 1.0, agent_net_shares=100)
        >>> print(f"Agent received: ${divs:.2f}")
    """
    # Get the actual dividend payment from the previous round (if any)
    # We're recording data for round N, and dividends were paid at end of round N-1
    dividends_received = 0.0
    used_payment_history = False

    if hasattr(agent, 'payment_history') and 'dividend' in agent.payment_history:
        dividend_payments = agent.payment_history['dividend']
        # Find dividend payments from the previous round (round_number - 1)
        # or the most recent if we're in round 0
        previous_round = max(0, round_number - 1)
        for payment in reversed(dividend_payments):
            if payment.round_number == previous_round:
                dividends_received += payment.amount
        dividends_received = round(dividends_received, 2)
        used_payment_history = True  # We checked payment_history, trust the result (even if $0)

    # Fallback for scenarios without payment history: calculate from positions
    # Note: This has timing issues in multi-stock scenarios where positions change
    if not used_payment_history and dividends != 0.0:
        if dividends_by_stock is not None:
            # Multi-stock: Calculate based on current positions (timing issue warning)
            for stock_id, dividend_per_share in dividends_by_stock.items():
                shares_in_stock = agent.positions.get(stock_id, 0)
                committed_in_stock = agent.committed_positions.get(stock_id, 0)
                borrowed_in_stock = agent.borrowed_positions.get(stock_id, 0)
                net_position = shares_in_stock + committed_in_stock - borrowed_in_stock
                dividends_received += net_position * dividend_per_share
            dividends_received = round(dividends_received, 2)
        else:
            # Single-stock: Use current positions
            dividends_received = round(agent_net_shares * dividends, 2)

    return dividends_received


def calculate_total_dividend_cash(
    agent_repository,
    round_number: int
) -> float:
    """
    Calculate total dividend cash paid across all agents from payment history.

    Aggregates actual dividend payments made in the previous round by summing
    payment_history['dividend'] entries for all agents.

    Args:
        agent_repository: Repository containing all agents
        round_number: Current simulation round

    Returns:
        Total dividend cash paid, rounded to 2 decimal places

    Example:
        >>> total = calculate_total_dividend_cash(agent_repo, 5)
        >>> print(f"Total dividends paid: ${total:.2f}")
    """
    # Dividends were paid at END of round_number-1
    previous_round = max(0, round_number - 1)
    total_cash_paid = 0.0

    for agent_id in agent_repository.get_all_agent_ids():
        agent = agent_repository.get_agent(agent_id)
        if hasattr(agent, 'payment_history') and 'dividend' in agent.payment_history:
            for payment in agent.payment_history['dividend']:
                if payment.round_number == previous_round:
                    total_cash_paid += payment.amount

    return round(total_cash_paid, 2)


def calculate_multi_stock_dividend_cash(
    agent_repository,
    round_number: int,
    stock_ids: list
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate total and per-stock dividend cash from payment history.

    For multi-stock scenarios, aggregates dividend payments and breaks them
    down by stock_id for detailed tracking.

    Args:
        agent_repository: Repository containing all agents
        round_number: Current simulation round
        stock_ids: List of stock IDs to track

    Returns:
        Tuple of (total_cash_paid, stock_cash_paid_dict) both rounded to 2 decimals

    Example:
        >>> total, per_stock = calculate_multi_stock_dividend_cash(repo, 5, ['AAPL', 'MSFT'])
        >>> print(f"Total: ${total:.2f}, AAPL: ${per_stock['AAPL']:.2f}")
    """
    # Dividends were paid at END of round_number-1, so look for those payments
    previous_round = max(0, round_number - 1)
    total_cash_paid_aggregate = 0.0
    stock_cash_paid = {stock_id: 0.0 for stock_id in stock_ids}

    # Sum actual payments from all agents' payment history, tracking per-stock breakdown
    for agent_id in agent_repository.get_all_agent_ids():
        agent = agent_repository.get_agent(agent_id)
        if hasattr(agent, 'payment_history') and 'dividend' in agent.payment_history:
            for payment in agent.payment_history['dividend']:
                if payment.round_number == previous_round:
                    # This payment was made in the previous round
                    total_cash_paid_aggregate += payment.amount

                    # Track per-stock breakdown using stock_id from payment
                    if payment.stock_id and payment.stock_id in stock_cash_paid:
                        stock_cash_paid[payment.stock_id] += payment.amount

    return (
        round(total_cash_paid_aggregate, 2),
        {stock_id: round(amount, 2) for stock_id, amount in stock_cash_paid.items()}
    )
