import sys, logging, types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

class _TestLoggingService:
    @staticmethod
    def get_logger(name):
        return logging.getLogger(name)

    @staticmethod
    def log_agent_state(*args, **kwargs):
        pass

    @staticmethod
    def log_validation_error(*args, **kwargs):
        pass

sys.modules.setdefault("services.logging_service", types.ModuleType("services.logging_service"))
sys.modules["services.logging_service"].LoggingService = _TestLoggingService

from agents.base_agent import BaseAgent
from agents.agent_manager.agent_repository import AgentRepository
from agents.agent_manager.services.borrowing_repository import BorrowingRepository
from agents.agents_api import TradeDecision
from market.state.sim_context import SimulationContext
from market.state.services.dividend_service import DividendPaymentProcessor

class DummyAgent(BaseAgent):
    def make_decision(self, market_state, history, round_number):
        return TradeDecision(orders=[], replace_decision="Cancel", reasoning="")

def test_short_sale_and_cover_tracks_borrow():
    context = SimulationContext(
        num_rounds=1,
        initial_price=100,
        fundamental_price=100,
        redemption_value=0,
        transaction_cost=0,
    )
    context.round_number = 1
    context.current_price = 100

    agent = DummyAgent("short", initial_cash=1000, initial_shares=0, allow_short_selling=True)
    borrow_repo = BorrowingRepository(total_lendable=100, logger=None)
    repo = AgentRepository([agent], logger=None, context=context, borrowing_repository=borrow_repo)

    # Agent commits shares to sell short
    result = repo.commit_shares("short", 5)
    assert result.success
    assert agent.borrowed_shares == 5
    assert agent.shares == 0

    # After trade execution, borrowed shares remain outstanding
    repo.release_resources("short", share_amount=5, return_borrowed=False)
    assert agent.borrowed_shares == 5
    assert agent.shares == 0
    assert borrow_repo.get_borrowed("short") == 5

    # Buying to cover releases borrowed shares
    repo.update_share_balance("short", 5)
    assert agent.borrowed_shares == 0
    assert agent.shares == 0
    assert borrow_repo.get_borrowed("short") == 0
    assert borrow_repo.available_shares == 100


def test_redemption_clears_outstanding_short():
    context = SimulationContext(
        num_rounds=1,
        initial_price=100,
        fundamental_price=100,
        redemption_value=100,
        transaction_cost=0,
    )
    context.round_number = 1
    context.current_price = 100

    agent = DummyAgent("short", initial_cash=1000, initial_shares=0, allow_short_selling=True)
    borrow_repo = BorrowingRepository(total_lendable=100, logger=None)
    repo = AgentRepository([agent], logger=None, context=context, borrowing_repository=borrow_repo)

    # Simulate a short sale of 5 shares at $100
    repo.commit_shares("short", 5)
    repo.release_resources("short", share_amount=5, return_borrowed=False)
    repo.update_account_balance("short", 500, account_type="main")

    processor = DividendPaymentProcessor(agent_repository=repo, logger=None)
    result = processor.process_redemption(100, round_number=1)

    assert result.success
    assert agent.borrowed_shares == 0
    assert borrow_repo.get_borrowed("short") == 0
    assert agent.shares == 0
    assert agent.cash == pytest.approx(1000)
