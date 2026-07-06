"""Regression tests for issue #89: cash commitment mismatch during matching.

Root cause: market orders hold a cash commitment while they are mid-matching
(state MATCHING; aggressive limits use LIMIT_MATCHING), and trades execute
inline during the matching phase, calling `sync_agent_orders` on both
counterparties. `is_active` originally excluded the two *_MATCHING states, so
when an agent with an in-flight market buy was the counterparty of another
trade, the invariant saw the commitment on the agent but no active order
carrying it and aborted the run ("Cash commitment mismatch for agent N").

Fixed by adding MATCHING/LIMIT_MATCHING to `is_active` (commit b3a8b35,
"is active = has commitment held"). These tests pin that behavior: removing
either state from `is_active` makes them fail exactly the way the original
runs crashed.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agents.agent_manager.agent_repository import AgentRepository
from agents.agent_manager.services.order_services import is_active
from market.orders.order import Order, OrderState


COMMITMENT_HOLDING_STATES = {
    OrderState.ACTIVE,
    OrderState.PENDING,
    OrderState.PARTIALLY_FILLED,
    OrderState.COMMITTED,
    OrderState.MATCHING,
    OrderState.LIMIT_MATCHING,
}


def _order(side, state, commitment, order_type="market", price=None):
    order = Order(agent_id="5", order_type=order_type, side=side, quantity=10,
                  round_placed=1, price=price)
    order.state = state
    if side == "buy":
        order.current_cash_commitment = commitment
    else:
        order.current_share_commitment = commitment
    return order


class _FakeAgent:
    """Minimal stand-in exposing only what sync_agent_orders touches."""

    def __init__(self, committed_cash):
        self.agent_id = "5"
        self.cash = 1000.0
        self.committed_cash = committed_cash
        self.dividend_cash = 0.0
        self.shares = 100
        self.committed_shares = 0
        self.committed_positions = {}
        self.synced = None

    def sync_orders(self, orders):
        self.synced = orders


def _make_repo(agent):
    repo = AgentRepository.__new__(AgentRepository)
    repo._agents = {agent.agent_id: agent}
    return repo


@pytest.mark.parametrize("state", sorted(COMMITMENT_HOLDING_STATES, key=lambda s: s.value))
def test_is_active_covers_every_commitment_holding_state(state):
    assert is_active(_order("buy", state, 290.0))


@pytest.mark.parametrize("state", [OrderState.INPUT, OrderState.VALIDATED,
                                   OrderState.FILLED, OrderState.CANCELLED])
def test_is_active_excludes_states_without_commitment(state):
    assert not is_active(_order("buy", state, 0.0))


def test_sync_counts_matching_market_buy_commitment():
    # The #89 crash shape: agent 5's market buy is mid-matching (MATCHING state,
    # cash committed) when a trade against one of its resting orders triggers
    # sync_agent_orders. The in-flight order must be counted or sync raises.
    agent = _FakeAgent(committed_cash=290.0)
    repo = _make_repo(agent)

    in_flight = _order("buy", OrderState.MATCHING, 290.0)
    repo.sync_agent_orders("5", [in_flight])  # must not raise

    assert agent.synced == [in_flight]


def test_sync_counts_limit_matching_buy_commitment():
    agent = _FakeAgent(committed_cash=576.0)
    repo = _make_repo(agent)

    in_flight = _order("buy", OrderState.LIMIT_MATCHING, 576.0,
                       order_type="limit", price=72.0)
    repo.sync_agent_orders("5", [in_flight])  # must not raise

    assert agent.synced == [in_flight]


def test_sync_counts_matching_sell_share_commitment():
    # A resting ACTIVE sell keeps the per-stock check non-vacuous: the check
    # only runs for stocks that appear via active sell orders, so without the
    # resting order a regressed is_active would empty the set and skip the
    # check entirely instead of failing it.
    agent = _FakeAgent(committed_cash=0.0)
    agent.committed_positions = {"DEFAULT_STOCK": 15}
    repo = _make_repo(agent)

    resting = _order("sell", OrderState.ACTIVE, 5, order_type="limit", price=72.0)
    in_flight = _order("sell", OrderState.MATCHING, 10)
    repo.sync_agent_orders("5", [resting, in_flight])  # must not raise

    assert agent.synced == [resting, in_flight]


def test_sync_still_raises_on_genuine_mismatch():
    # The invariant must stay loud when the commitment truly has no owner.
    agent = _FakeAgent(committed_cash=290.0)
    repo = _make_repo(agent)

    with pytest.raises(ValueError, match="Cash commitment mismatch"):
        repo.sync_agent_orders("5", [])
