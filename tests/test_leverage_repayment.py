"""Regression tests for issue #92: leverage margin-call trades must repay
borrowed cash exactly once.

Repayment happens in the AUTOMATIC DEBT REPAYMENT block of
AgentRepository.update_agent_position_after_trade. The matching engine must
NOT run a second repayment pass over margin-call trades — doing so applied
the same sale proceeds twice in the partial-repayment case (debt understated,
lending pool double-credited).
"""
import sys, logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision
from agents.agent_manager.agent_repository import AgentRepository
from agents.agent_manager.services.position_services import PositionChange
from agents.agent_manager.services.cash_lending_repository import CashLendingRepository


class DummyAgent(BaseAgent):
    def make_decision(self, market_state, history, round_number):
        return TradeDecision(orders=[], replace_decision="Cancel", reasoning="")


class StubContext:
    round_number = 1

    def __init__(self):
        self.repayments = []

    def record_leverage_cash_repaid(self, amount, round_number):
        self.repayments.append(amount)


def _make_leveraged_agent(initial_cash, debt, pool_cash=100_000):
    agent = DummyAgent("lev_agent", initial_cash=initial_cash, initial_shares=0)
    lending_repo = CashLendingRepository(total_lendable_cash=pool_cash)
    allocated = lending_repo.allocate_cash(agent.agent_id, debt)
    assert allocated == debt
    agent.borrowed_cash = debt
    agent.cash_lending_repo = lending_repo
    context = StubContext()
    repo = AgentRepository([agent], logger=logging.getLogger("test"), context=context)
    return agent, lending_repo, context, repo


def test_partial_repayment_applied_exactly_once():
    """Sale proceeds < debt: debt drops by proceeds (not 2x) and the lending
    pool is credited the proceeds exactly once."""
    agent, lending_repo, context, repo = _make_leveraged_agent(
        initial_cash=1000, debt=1000
    )
    proceeds = 300.0

    repo.update_agent_position_after_trade(
        agent.agent_id,
        PositionChange(cash_change=proceeds, shares_change=0),
    )

    assert agent.borrowed_cash == 700, (
        f"Debt should drop by proceeds exactly once: expected 700, got {agent.borrowed_cash}"
    )
    # Proceeds arrive then are immediately used for repayment: net cash unchanged
    assert agent.cash == 1000
    assert lending_repo.borrowed[agent.agent_id] == 700
    assert lending_repo.available_cash == 100_000 - 700
    assert context.repayments == [proceeds]


def test_full_repayment_keeps_excess_proceeds():
    """Sale proceeds > debt: debt cleared, agent keeps the remainder."""
    agent, lending_repo, context, repo = _make_leveraged_agent(
        initial_cash=1000, debt=200
    )

    repo.update_agent_position_after_trade(
        agent.agent_id,
        PositionChange(cash_change=300.0, shares_change=0),
    )

    assert agent.borrowed_cash == 0
    assert agent.cash == 1100  # +300 proceeds, -200 repayment
    assert agent.agent_id not in lending_repo.borrowed
    assert context.repayments == [200.0]


def test_engine_has_no_second_repayment_pass():
    """The matching engine must not re-apply margin-call sale proceeds to
    borrowed cash — the trade pipeline's auto-repay already did it (issue #92)."""
    engine_source = (
        Path(__file__).resolve().parents[1]
        / "src" / "market" / "engine" / "match_engine.py"
    ).read_text()
    assert "_process_leverage_margin_call_repayments" not in engine_source
