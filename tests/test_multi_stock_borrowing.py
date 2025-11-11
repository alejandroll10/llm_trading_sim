"""
Test multi-stock borrowing repository isolation (Issue #48)

This test file verifies that borrowing shares from one stock does not
affect the lending capacity of other stocks in multi-stock scenarios.
"""
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


class DummyAgent(BaseAgent):
    def make_decision(self, market_state, history, round_number):
        return TradeDecision(orders=[], replace_decision="Cancel", reasoning="")


def test_multi_stock_borrowing_isolation():
    """Test that borrowing from one stock doesn't affect another stock's pool"""
    context = SimulationContext(
        num_rounds=1,
        initial_price=100,
        fundamental_price=100,
        redemption_value=0,
        transaction_cost=0,
    )
    context.round_number = 1
    context.current_price = 100

    # Create agent with sufficient cash for margin requirements
    # Short selling requires 50% margin by default: to borrow 500 shares @ $100 = $50k value, need $25k cash
    # This test focuses on borrowing pool isolation, not margin validation
    agent = DummyAgent(
        "trader1",
        initial_cash=100000,  # Sufficient for margin requirements
        initial_shares=0,
        allow_short_selling=True
    )
    agent.positions = {"TECH_A": 0, "PHARMA_B": 0, "ENERGY_C": 0}
    agent.borrowed_positions = {"TECH_A": 0, "PHARMA_B": 0, "ENERGY_C": 0}

    # Create separate borrowing repositories for each stock
    borrowing_repos = {
        "TECH_A": BorrowingRepository(total_lendable=1000, logger=None),
        "PHARMA_B": BorrowingRepository(total_lendable=2000, logger=None),
        "ENERGY_C": BorrowingRepository(total_lendable=500, logger=None)
    }

    # Create AgentRepository with multi-stock borrowing repositories
    repo = AgentRepository(
        [agent],
        logger=None,
        context=context,
        borrowing_repositories=borrowing_repos
    )

    # Verify initial state
    assert borrowing_repos["TECH_A"].available_shares == 1000
    assert borrowing_repos["PHARMA_B"].available_shares == 2000
    assert borrowing_repos["ENERGY_C"].available_shares == 500

    # CRITICAL TEST: Borrow from TECH_A
    result = repo.commit_shares("trader1", 500, stock_id="TECH_A")
    assert result.success
    assert agent.borrowed_positions["TECH_A"] == 500

    # Verify TECH_A pool decreased
    assert borrowing_repos["TECH_A"].available_shares == 500
    assert borrowing_repos["TECH_A"].get_borrowed("trader1") == 500

    # CRITICAL TEST: PHARMA_B and ENERGY_C pools should be UNCHANGED
    assert borrowing_repos["PHARMA_B"].available_shares == 2000, \
        "PHARMA_B pool should not be affected by TECH_A borrowing"
    assert borrowing_repos["ENERGY_C"].available_shares == 500, \
        "ENERGY_C pool should not be affected by TECH_A borrowing"

    # Borrow from PHARMA_B
    result = repo.commit_shares("trader1", 1500, stock_id="PHARMA_B")
    assert result.success
    assert agent.borrowed_positions["PHARMA_B"] == 1500

    # Verify PHARMA_B pool decreased
    assert borrowing_repos["PHARMA_B"].available_shares == 500
    assert borrowing_repos["PHARMA_B"].get_borrowed("trader1") == 1500

    # Verify TECH_A and ENERGY_C pools remain independent
    assert borrowing_repos["TECH_A"].available_shares == 500, \
        "TECH_A pool should not be affected by PHARMA_B borrowing"
    assert borrowing_repos["ENERGY_C"].available_shares == 500, \
        "ENERGY_C pool should not be affected by PHARMA_B borrowing"


def test_multi_stock_borrowing_release():
    """Test that releasing borrowed shares returns to the correct pool"""
    context = SimulationContext(
        num_rounds=1,
        initial_price=100,
        fundamental_price=100,
        redemption_value=0,
        transaction_cost=0,
    )
    context.round_number = 1
    context.current_price = 100

    # Sufficient cash for margin: 300 TECH_A + 800 PHARMA_B = 1100 shares @ $100 = $110k value → need $55k margin
    agent = DummyAgent(
        "trader1",
        initial_cash=100000,  # Sufficient for margin requirements
        initial_shares=0,
        allow_short_selling=True
    )
    agent.positions = {"TECH_A": 0, "PHARMA_B": 0}
    agent.borrowed_positions = {"TECH_A": 0, "PHARMA_B": 0}

    borrowing_repos = {
        "TECH_A": BorrowingRepository(total_lendable=1000, logger=None),
        "PHARMA_B": BorrowingRepository(total_lendable=2000, logger=None),
    }

    repo = AgentRepository(
        [agent],
        logger=None,
        context=context,
        borrowing_repositories=borrowing_repos
    )

    # Borrow from both stocks
    repo.commit_shares("trader1", 300, stock_id="TECH_A")
    repo.commit_shares("trader1", 800, stock_id="PHARMA_B")

    assert borrowing_repos["TECH_A"].available_shares == 700
    assert borrowing_repos["PHARMA_B"].available_shares == 1200

    # Cover TECH_A position by buying shares
    repo.update_share_balance("trader1", 300, stock_id="TECH_A")

    # CRITICAL TEST: Shares should return to TECH_A pool only
    assert borrowing_repos["TECH_A"].available_shares == 1000, \
        "TECH_A pool should be fully replenished"
    assert borrowing_repos["PHARMA_B"].available_shares == 1200, \
        "PHARMA_B pool should be unchanged"
    assert agent.borrowed_positions["TECH_A"] == 0
    assert agent.borrowed_positions["PHARMA_B"] == 800

    # Cover PHARMA_B position
    repo.update_share_balance("trader1", 800, stock_id="PHARMA_B")

    # Verify PHARMA_B pool replenished
    assert borrowing_repos["PHARMA_B"].available_shares == 2000
    assert agent.borrowed_positions["PHARMA_B"] == 0


def test_multi_stock_borrowing_capacity_limits():
    """Test that each stock has independent borrowing capacity"""
    context = SimulationContext(
        num_rounds=1,
        initial_price=100,
        fundamental_price=100,
        redemption_value=0,
        transaction_cost=0,
    )
    context.round_number = 1
    context.current_price = 100

    # Sufficient cash for borrowing 1000 shares @ $100 = $100k value → need $50k margin
    agent = DummyAgent(
        "trader1",
        initial_cash=100000,  # Sufficient for margin requirements
        initial_shares=0,
        allow_short_selling=True
    )
    agent.positions = {"TECH_A": 0, "PHARMA_B": 0}
    agent.borrowed_positions = {"TECH_A": 0, "PHARMA_B": 0}

    borrowing_repos = {
        "TECH_A": BorrowingRepository(total_lendable=100, logger=None),  # Small pool
        "PHARMA_B": BorrowingRepository(total_lendable=5000, logger=None),  # Large pool
    }

    repo = AgentRepository(
        [agent],
        logger=None,
        context=context,
        borrowing_repositories=borrowing_repos
    )

    # Try to borrow more than TECH_A capacity
    result = repo.commit_shares("trader1", 200, stock_id="TECH_A")
    assert not result.success, "Should fail - not enough TECH_A shares"
    assert agent.borrowed_positions["TECH_A"] == 0

    # CRITICAL TEST: PHARMA_B should still have full capacity
    result = repo.commit_shares("trader1", 1000, stock_id="PHARMA_B")
    assert result.success, "Should succeed - PHARMA_B has independent large pool"
    assert agent.borrowed_positions["PHARMA_B"] == 1000
    assert borrowing_repos["PHARMA_B"].available_shares == 4000


def test_multi_stock_redemption_releases_all_pools():
    """Test that redemption releases borrowed shares from all stocks"""
    context = SimulationContext(
        num_rounds=1,
        initial_price=100,
        fundamental_price=100,
        redemption_value=100,
        transaction_cost=0,
    )
    context.round_number = 1
    context.current_price = 100

    agent = DummyAgent(
        "trader1",
        initial_cash=200000,  # Enough for margin on all three stocks
        initial_shares=0,
        allow_short_selling=True
    )
    agent.positions = {"TECH_A": 0, "PHARMA_B": 0, "ENERGY_C": 0}
    agent.borrowed_positions = {"TECH_A": 0, "PHARMA_B": 0, "ENERGY_C": 0}

    borrowing_repos = {
        "TECH_A": BorrowingRepository(total_lendable=1000, logger=None),
        "PHARMA_B": BorrowingRepository(total_lendable=2000, logger=None),
        "ENERGY_C": BorrowingRepository(total_lendable=500, logger=None)
    }

    repo = AgentRepository(
        [agent],
        logger=None,
        context=context,
        borrowing_repositories=borrowing_repos
    )

    # Borrow from all three stocks
    repo.commit_shares("trader1", 300, stock_id="TECH_A")
    repo.commit_shares("trader1", 800, stock_id="PHARMA_B")
    repo.commit_shares("trader1", 200, stock_id="ENERGY_C")

    # Simulate redemption
    repo.redeem_all_shares("trader1")

    # CRITICAL TEST: All pools should be fully replenished
    assert borrowing_repos["TECH_A"].available_shares == 1000
    assert borrowing_repos["PHARMA_B"].available_shares == 2000
    assert borrowing_repos["ENERGY_C"].available_shares == 500
    assert agent.borrowed_positions["TECH_A"] == 0
    assert agent.borrowed_positions["PHARMA_B"] == 0
    assert agent.borrowed_positions["ENERGY_C"] == 0


def test_single_stock_backward_compatibility():
    """Test that single-stock mode still works with single borrowing repository"""
    context = SimulationContext(
        num_rounds=1,
        initial_price=100,
        fundamental_price=100,
        redemption_value=0,
        transaction_cost=0,
    )
    context.round_number = 1
    context.current_price = 100

    agent = DummyAgent(
        "trader1",
        initial_cash=50000,  # Enough for margin requirements
        initial_shares=0,
        allow_short_selling=True
    )

    # Single borrowing repository (old style)
    borrow_repo = BorrowingRepository(total_lendable=1000, logger=None)

    # Create AgentRepository with single repository (backward compatible)
    repo = AgentRepository(
        [agent],
        logger=None,
        context=context,
        borrowing_repository=borrow_repo  # Single repo, not dict
    )

    # Should work with DEFAULT_STOCK
    result = repo.commit_shares("trader1", 500, stock_id="DEFAULT_STOCK")
    assert result.success
    assert agent.borrowed_shares == 500
    assert borrow_repo.available_shares == 500
    assert borrow_repo.get_borrowed("trader1") == 500


def test_partial_borrows_per_stock():
    """Test that partial borrows work independently per stock"""
    context = SimulationContext(
        num_rounds=1,
        initial_price=100,
        fundamental_price=100,
        redemption_value=0,
        transaction_cost=0,
    )
    context.round_number = 1
    context.current_price = 100

    agent = DummyAgent(
        "trader1",
        initial_cash=20000,  # Enough for margin on partial borrows
        initial_shares=0,
        allow_short_selling=True
    )
    agent.positions = {"TECH_A": 0, "PHARMA_B": 0}
    agent.borrowed_positions = {"TECH_A": 0, "PHARMA_B": 0}

    # Small pools with partial borrows enabled
    borrowing_repos = {
        "TECH_A": BorrowingRepository(total_lendable=100, allow_partial_borrows=True, logger=None),
        "PHARMA_B": BorrowingRepository(total_lendable=50, allow_partial_borrows=True, logger=None),
    }

    repo = AgentRepository(
        [agent],
        logger=None,
        context=context,
        borrowing_repositories=borrowing_repos
    )

    # Request more than available from TECH_A - should get partial fill
    result = repo.commit_shares("trader1", 200, stock_id="TECH_A")
    assert result.success
    assert result.partial_fill
    assert result.committed_amount == 100  # Got all available
    assert agent.borrowed_positions["TECH_A"] == 100

    # CRITICAL TEST: PHARMA_B should still have full capacity
    result = repo.commit_shares("trader1", 30, stock_id="PHARMA_B")
    assert result.success
    assert not result.partial_fill  # Full fill available
    assert agent.borrowed_positions["PHARMA_B"] == 30
    assert borrowing_repos["PHARMA_B"].available_shares == 20


def test_margin_requirement_rejection():
    """Test that system gracefully rejects borrows when margin is insufficient

    This is expected LLM behavior - they may request too much.
    System should:
    1. Log a warning (already done via LoggingService.log_validation_error)
    2. Return CommitmentResult(success=False) gracefully
    3. NOT crash or raise unhandled exceptions
    """
    context = SimulationContext(
        num_rounds=1,
        initial_price=100,
        fundamental_price=100,
        redemption_value=0,
        transaction_cost=0,
    )
    context.round_number = 1
    context.current_price = 100

    # Agent with INSUFFICIENT cash for margin requirements
    agent = DummyAgent(
        "poor_trader",
        initial_cash=5000,  # Not enough for 500 shares @ $100 (needs $25k for 50% margin)
        initial_shares=0,
        allow_short_selling=True
    )
    agent.positions = {"TECH_A": 0}
    agent.borrowed_positions = {"TECH_A": 0}

    borrowing_repos = {
        "TECH_A": BorrowingRepository(total_lendable=10000, logger=None),  # Plenty available
    }

    repo = AgentRepository(
        [agent],
        logger=None,
        context=context,
        borrowing_repositories=borrowing_repos
    )

    # Try to borrow more than margin allows
    result = repo.commit_shares("poor_trader", 500, stock_id="TECH_A")

    # CRITICAL: Should gracefully reject, not crash
    assert not result.success, "Should reject when margin insufficient"
    assert "Margin requirement not met" in result.message

    # Verify no shares were borrowed
    assert agent.borrowed_positions["TECH_A"] == 0
    assert borrowing_repos["TECH_A"].get_borrowed("poor_trader") == 0
    assert borrowing_repos["TECH_A"].available_shares == 10000  # Pool unchanged

    # Verify this rejection doesn't affect ability to borrow from other stocks
    # (if margin requirements are met for those stocks)
    agent.positions["PHARMA_B"] = 0
    agent.borrowed_positions["PHARMA_B"] = 0

    # Add PHARMA_B to borrowing repos and update repo to use it
    borrowing_repos["PHARMA_B"] = BorrowingRepository(total_lendable=5000, logger=None)
    repo.borrowing_repositories = borrowing_repos

    # Give agent more cash for margin on PHARMA_B
    agent.cash = 50000
    result2 = repo.commit_shares("poor_trader", 100, stock_id="PHARMA_B")
    assert result2.success, "Should succeed with sufficient margin, independent of TECH_A rejection"
    assert borrowing_repos["PHARMA_B"].available_shares == 4900

    # TECH_A pool should still be untouched
    assert borrowing_repos["TECH_A"].available_shares == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
