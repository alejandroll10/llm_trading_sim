#!/usr/bin/env python3
"""
Standalone validation script for multi-stock margin call implementation.
This script tests the key functionality without requiring pytest.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Mock the logging service
import types
import logging

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


def test_total_borrowed_shares():
    """Test 1: Verify total_borrowed_shares sums correctly"""
    print("Test 1: total_borrowed_shares property...")

    agent = DummyAgent(
        "test_agent",
        initial_cash=10000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5
    )

    agent.borrowed_positions["STOCK_A"] = 10
    agent.borrowed_positions["STOCK_B"] = 20
    agent.borrowed_positions["STOCK_C"] = 15

    assert agent.total_borrowed_shares == 45, f"Expected 45, got {agent.total_borrowed_shares}"
    assert agent.borrowed_shares == 0, f"Expected 0 for DEFAULT_STOCK, got {agent.borrowed_shares}"

    print("  ✓ PASSED")


def test_portfolio_margin_status():
    """Test 2: Verify portfolio margin status calculation"""
    print("Test 2: Portfolio margin status calculation...")

    agent = DummyAgent(
        "test_agent",
        initial_cash=10000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )

    agent.borrowed_positions["STOCK_A"] = 10  # Value: 1000
    agent.borrowed_positions["STOCK_B"] = 5   # Value: 1000

    prices = {"STOCK_A": 100, "STOCK_B": 200}
    status = agent.get_portfolio_margin_status(prices)

    assert status['borrowed_value'] == 2000, f"Expected 2000, got {status['borrowed_value']}"
    assert status['collateral'] == 10000, f"Expected 10000, got {status['collateral']}"
    assert status['max_borrowable_value'] == 20000, f"Expected 20000, got {status['max_borrowable_value']}"
    assert not status['is_margin_violated'], "Should not be violated"

    print("  ✓ PASSED")


def test_margin_violation_detection():
    """Test 3: Verify margin violations are detected"""
    print("Test 3: Margin violation detection...")

    agent = DummyAgent(
        "test_agent",
        initial_cash=1000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )

    agent.borrowed_positions["STOCK_A"] = 15  # Value: 1500
    agent.borrowed_positions["STOCK_B"] = 10  # Value: 2000

    prices = {"STOCK_A": 100, "STOCK_B": 200}
    status = agent.get_portfolio_margin_status(prices)

    # Total borrowed: 3500, Max: 2000, Excess: 1500
    assert status['borrowed_value'] == 3500, f"Expected 3500, got {status['borrowed_value']}"
    assert status['max_borrowable_value'] == 2000, f"Expected 2000, got {status['max_borrowable_value']}"
    assert status['is_margin_violated'], "Should be violated"
    assert status['excess_borrowed_value'] == 1500, f"Expected 1500, got {status['excess_borrowed_value']}"

    print("  ✓ PASSED")


def test_margin_call_execution():
    """Test 4: Verify margin call forces covering"""
    print("Test 4: Margin call execution...")

    agent = DummyAgent(
        "test_agent",
        initial_cash=10000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="wealth"  # Use wealth-based margin for more realistic behavior
    )

    # Create violation with smaller amount to ensure covering resolves it
    # Use proper accumulator pattern
    agent.borrowed_positions["STOCK_A"] = 80  # Value: 8000
    agent.borrowed_shares += 80  # Accumulator
    agent.borrowed_positions["STOCK_B"] = 30   # Value: 6000
    agent.borrowed_shares += 30  # Accumulator

    prices = {"STOCK_A": 100, "STOCK_B": 200}

    # Initial status: borrowed = 14000
    # Net position = 0 - 14000 = -14000
    # Collateral = 10000 + (-14000) = -4000 (negative!)
    # This is a severe violation

    initial_borrowed_total = agent.total_borrowed_shares
    initial_cash = agent.cash

    agent.handle_multi_stock_margin_call(prices, round_number=1)

    # Should have covered some shares
    assert agent.total_borrowed_shares < initial_borrowed_total, "Should have covered shares"
    assert agent.cash < initial_cash, "Cash should have been used"

    # With wealth-based margin, buying shares increases collateral
    # So the situation should improve
    status = agent.get_portfolio_margin_status(prices)

    # The excess should be significantly reduced (though may not be zero due to cash constraints)
    initial_status = {"borrowed_value": 14000}
    assert agent.total_borrowed_shares < initial_borrowed_total * 0.9, "Should have covered significant amount"

    print("  ✓ PASSED")


def test_update_wealth_integration():
    """Test 5: Verify update_wealth triggers margin calls"""
    print("Test 5: update_wealth integration...")

    agent = DummyAgent(
        "test_agent",
        initial_cash=5000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )
    agent.last_update_round = 1

    # Create violation (use accumulator pattern for real-world simulation)
    agent.borrowed_positions["STOCK_A"] = 100  # Value: 10000
    agent.borrowed_shares += 100  # Accumulator
    agent.borrowed_positions["STOCK_B"] = 25   # Value: 5000
    agent.borrowed_shares += 25  # Accumulator

    prices = {"STOCK_A": 100, "STOCK_B": 200}

    # Total: 15000, Max: 10000, Excess: 5000
    initial_borrowed = agent.total_borrowed_shares

    agent.update_wealth(prices)

    assert agent.total_borrowed_shares < initial_borrowed, "update_wealth should trigger margin call"

    print("  ✓ PASSED")


def test_backward_compatibility():
    """Test 6: Verify single-stock scenarios still work"""
    print("Test 6: Backward compatibility with single-stock...")

    agent = DummyAgent(
        "test_agent",
        initial_cash=1000,
        initial_shares=0,
        allow_short_selling=True,
        margin_requirement=0.5,
        margin_base="cash"
    )
    agent.last_update_round = 1

    # Use single-stock API
    agent.borrowed_shares = 50  # Value: 5000

    price = 100.0

    # Max: 2000, Borrowed: 5000, Violation!
    agent.update_wealth(price)

    assert agent.borrowed_shares < 50, "Single-stock margin call should have triggered"

    print("  ✓ PASSED")


def main():
    print("\n" + "="*60)
    print("Multi-Stock Margin Call Implementation Validation")
    print("="*60 + "\n")

    tests = [
        test_total_borrowed_shares,
        test_portfolio_margin_status,
        test_margin_violation_detection,
        test_margin_call_execution,
        test_update_wealth_integration,
        test_backward_compatibility,
    ]

    failed = 0
    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "="*60)
    if failed == 0:
        print("✓ All tests PASSED!")
    else:
        print(f"✗ {failed} test(s) FAILED")
    print("="*60 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
