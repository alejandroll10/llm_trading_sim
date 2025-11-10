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

    @staticmethod
    def log_margin_call(*args, **kwargs):
        pass

sys.modules.setdefault("services.logging_service", types.ModuleType("services.logging_service"))
sys.modules["services.logging_service"].LoggingService = _TestLoggingService

from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision

class DummyAgent(BaseAgent):
    def make_decision(self, market_state, history, round_number):
        return TradeDecision(orders=[], replace_decision="Cancel", reasoning="")


def test_total_borrowed_shares_sums_across_stocks():
    """Test that total_borrowed_shares correctly sums borrowed positions across all stocks"""
    agent = DummyAgent(
        "test_agent",
        initial_cash=10000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5
    )

    # Borrow shares from multiple stocks
    agent.borrowed_positions["STOCK_A"] = 10
    agent.borrowed_positions["STOCK_B"] = 20
    agent.borrowed_positions["STOCK_C"] = 15

    assert agent.total_borrowed_shares == 45
    # Verify backward compatibility - borrowed_shares still returns DEFAULT_STOCK
    assert agent.borrowed_shares == 0


def test_portfolio_margin_status_no_borrowing():
    """Test portfolio margin status when no shares are borrowed"""
    agent = DummyAgent(
        "test_agent",
        initial_cash=10000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5
    )

    prices = {"STOCK_A": 100, "STOCK_B": 200}
    status = agent.get_portfolio_margin_status(prices)

    assert status['borrowed_value'] == 0
    assert status['is_margin_violated'] == False
    assert status['margin_ratio'] == float('inf')


def test_portfolio_margin_status_with_borrowing():
    """Test portfolio margin status calculation with borrowed shares"""
    agent = DummyAgent(
        "test_agent",
        initial_cash=10000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )

    # Borrow shares from two stocks
    agent.borrowed_positions["STOCK_A"] = 10  # Value: 10 * 100 = 1000
    agent.borrowed_positions["STOCK_B"] = 5   # Value: 5 * 200 = 1000

    prices = {"STOCK_A": 100, "STOCK_B": 200}
    status = agent.get_portfolio_margin_status(prices)

    assert status['borrowed_value'] == 2000  # 1000 + 1000
    assert status['collateral'] == 10000  # Cash only
    assert status['max_borrowable_value'] == 20000  # 10000 / 0.5
    assert status['is_margin_violated'] == False
    assert status['excess_borrowed_value'] == 0


def test_portfolio_margin_violation_detection():
    """Test that margin violations are correctly detected"""
    agent = DummyAgent(
        "test_agent",
        initial_cash=1000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )

    # Borrow shares worth more than allowed
    agent.borrowed_positions["STOCK_A"] = 15  # Value: 15 * 100 = 1500
    agent.borrowed_positions["STOCK_B"] = 10  # Value: 10 * 200 = 2000

    prices = {"STOCK_A": 100, "STOCK_B": 200}
    status = agent.get_portfolio_margin_status(prices)

    # Total borrowed value: 3500
    # Max borrowable: 1000 / 0.5 = 2000
    # Excess: 3500 - 2000 = 1500
    assert status['borrowed_value'] == 3500
    assert status['max_borrowable_value'] == 2000
    assert status['is_margin_violated'] == True
    assert status['excess_borrowed_value'] == 1500


def test_multi_stock_margin_call_no_violation():
    """Test that no margin call occurs when requirements are met"""
    agent = DummyAgent(
        "test_agent",
        initial_cash=10000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )

    agent.borrowed_positions["STOCK_A"] = 10
    agent.borrowed_positions["STOCK_B"] = 5
    initial_borrowed_a = agent.borrowed_positions["STOCK_A"]
    initial_borrowed_b = agent.borrowed_positions["STOCK_B"]

    prices = {"STOCK_A": 100, "STOCK_B": 200}
    agent.handle_multi_stock_margin_call(prices, round_number=1)

    # No changes should occur
    assert agent.borrowed_positions["STOCK_A"] == initial_borrowed_a
    assert agent.borrowed_positions["STOCK_B"] == initial_borrowed_b


def test_multi_stock_margin_call_forces_cover():
    """Test that margin call forces buy-to-cover when violated"""
    agent = DummyAgent(
        "test_agent",
        initial_cash=10000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )

    # Create a margin violation
    agent.borrowed_positions["STOCK_A"] = 150  # Value: 15000
    agent.borrowed_positions["STOCK_B"] = 50   # Value: 10000

    prices = {"STOCK_A": 100, "STOCK_B": 200}

    # Total borrowed value: 25000
    # Max borrowable: 10000 / 0.5 = 20000
    # Excess: 5000
    # This should trigger buy-to-cover

    initial_cash = agent.cash
    agent.handle_multi_stock_margin_call(prices, round_number=1)

    # Verify shares were covered
    assert agent.borrowed_positions["STOCK_A"] < 150
    assert agent.borrowed_positions["STOCK_B"] < 50

    # Verify cash was reduced (to buy shares)
    assert agent.cash < initial_cash

    # Verify margin is now satisfied
    status = agent.get_portfolio_margin_status(prices)
    # After covering, should be close to meeting requirements (may not be exact due to rounding)
    assert status['excess_borrowed_value'] < 100  # Small tolerance for rounding


def test_multi_stock_margin_call_proportional_covering():
    """Test that margin call covers positions proportionally"""
    agent = DummyAgent(
        "test_agent",
        initial_cash=10000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )

    # STOCK_A: 100 shares at $100 = $10,000 (66.67% of borrowed value)
    # STOCK_B: 25 shares at $200 = $5,000  (33.33% of borrowed value)
    agent.borrowed_positions["STOCK_A"] = 100
    agent.borrowed_positions["STOCK_B"] = 25

    prices = {"STOCK_A": 100, "STOCK_B": 200}

    # Total borrowed: 15000
    # Max borrowable: 10000 / 0.5 = 20000
    # Actually this is not violated, let me adjust...

    # Change to create violation:
    agent.borrowed_positions["STOCK_A"] = 150  # Value: 15000
    agent.borrowed_positions["STOCK_B"] = 62   # Value: 12400

    # Total borrowed: 27400
    # Max borrowable: 20000
    # Excess: 7400

    initial_borrowed_a = agent.borrowed_positions["STOCK_A"]
    initial_borrowed_b = agent.borrowed_positions["STOCK_B"]

    agent.handle_multi_stock_margin_call(prices, round_number=1)

    # Calculate how much was covered from each stock
    covered_a = initial_borrowed_a - agent.borrowed_positions["STOCK_A"]
    covered_b = initial_borrowed_b - agent.borrowed_positions["STOCK_B"]

    # Both stocks should have some covering (proportional)
    assert covered_a > 0
    assert covered_b > 0

    # The ratio of covered values should be roughly proportional to borrowed values
    # STOCK_A is roughly 55% of total borrowed value (15000/27400)
    # STOCK_B is roughly 45% of total borrowed value (12400/27400)
    value_covered_a = covered_a * prices["STOCK_A"]
    value_covered_b = covered_b * prices["STOCK_B"]
    total_covered_value = value_covered_a + value_covered_b

    proportion_a = value_covered_a / total_covered_value if total_covered_value > 0 else 0
    proportion_b = value_covered_b / total_covered_value if total_covered_value > 0 else 0

    # Check proportions are roughly correct (within 10% tolerance)
    assert abs(proportion_a - 0.55) < 0.1
    assert abs(proportion_b - 0.45) < 0.1


def test_update_wealth_triggers_multi_stock_margin_call():
    """Test that update_wealth automatically triggers margin calls in multi-stock scenarios"""
    agent = DummyAgent(
        "test_agent",
        initial_cash=5000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )
    agent.last_update_round = 1

    # Create a margin violation
    agent.borrowed_positions["STOCK_A"] = 100  # Value: 10000
    agent.borrowed_positions["STOCK_B"] = 25   # Value: 5000

    prices = {"STOCK_A": 100, "STOCK_B": 200}

    # Total borrowed: 15000
    # Max borrowable: 5000 / 0.5 = 10000
    # Excess: 5000 - violation!

    initial_borrowed_total = agent.total_borrowed_shares

    # Update wealth should trigger margin call
    agent.update_wealth(prices)

    # Verify margin call was triggered (some shares were covered)
    assert agent.total_borrowed_shares < initial_borrowed_total


def test_backward_compatibility_single_stock():
    """Test that single-stock margin calls still work (backward compatibility)"""
    agent = DummyAgent(
        "test_agent",
        initial_cash=1000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )
    agent.last_update_round = 1

    # Use old single-stock API
    agent.borrowed_shares = 50  # Value: 5000 at price 100

    price = 100.0

    # Max borrowable: 1000 / 0.5 = 2000
    # Current borrowed value: 5000
    # Violation!

    agent.update_wealth(price)

    # Should have triggered margin call
    assert agent.borrowed_shares < 50


def test_wealth_based_margin_calculation():
    """Test margin calculation when margin_base is 'wealth' instead of 'cash'"""
    agent = DummyAgent(
        "test_agent",
        initial_cash=1000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="wealth"  # Use total wealth as collateral
    )

    # Agent has long positions in STOCK_A and short in STOCK_B
    agent.positions["STOCK_A"] = 50  # Value: 5000
    agent.borrowed_positions["STOCK_B"] = 20  # Value: 4000

    prices = {"STOCK_A": 100, "STOCK_B": 200}

    status = agent.get_portfolio_margin_status(prices)

    # Net position value: 50*100 - 20*200 = 5000 - 4000 = 1000
    # Collateral: cash + net_position_value = 1000 + 1000 = 2000
    # Max borrowable: 2000 / 0.5 = 4000
    # Borrowed value: 4000
    # Should be right at the limit (not violated)

    assert status['collateral'] == 2000
    assert status['borrowed_value'] == 4000
    assert status['max_borrowable_value'] == 4000
    assert status['is_margin_violated'] == False


def test_margin_call_with_insufficient_cash():
    """Test margin call behavior when agent doesn't have enough cash to cover"""
    agent = DummyAgent(
        "test_agent",
        initial_cash=1000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )

    # Create a large margin violation
    agent.borrowed_positions["STOCK_A"] = 200  # Value: 20000

    prices = {"STOCK_A": 100}

    # Max borrowable: 1000 / 0.5 = 2000
    # Borrowed value: 20000
    # Excess: 18000
    # To cover excess would need $18000 but only have $1000

    agent.handle_multi_stock_margin_call(prices, round_number=1)

    # Should cover as much as possible with available cash
    # Cash might go negative (this models margin call forcing liquidation)
    # The important thing is that some covering occurred
    assert agent.borrowed_positions["STOCK_A"] < 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
