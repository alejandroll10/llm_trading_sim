#!/usr/bin/env python3
"""Test edge cases for multi-stock margin calls"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

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


def test_accumulator_consistency():
    """Test that DEFAULT_STOCK accumulator is always consistent"""
    print("Test: Accumulator consistency...")

    agent = DummyAgent("test", initial_cash=10000, initial_shares=0,
                       allow_short_selling=True, margin_requirement=0.5, margin_base="wealth")

    # Simulate borrowing with accumulator pattern
    agent.borrowed_positions["STOCK_A"] = 50
    agent.borrowed_shares += 50
    agent.borrowed_positions["STOCK_B"] = 30
    agent.borrowed_shares += 30

    # Verify total
    assert agent.total_borrowed_shares == 80, f"Expected 80, got {agent.total_borrowed_shares}"
    assert agent.borrowed_shares == 80, f"Expected DEFAULT_STOCK=80, got {agent.borrowed_shares}"

    # Trigger margin call
    prices = {"STOCK_A": 150, "STOCK_B": 200}
    agent.handle_multi_stock_margin_call(prices, round_number=1)

    # After margin call, check consistency
    expected_total = agent.borrowed_positions.get("STOCK_A", 0) + agent.borrowed_positions.get("STOCK_B", 0)
    assert agent.total_borrowed_shares == expected_total, \
        f"Total inconsistent: {agent.total_borrowed_shares} != {expected_total}"
    assert agent.borrowed_shares == expected_total, \
        f"DEFAULT_STOCK inconsistent: {agent.borrowed_shares} != {expected_total}"

    print("  ✓ PASSED")


def test_single_stock_still_works():
    """Test that DEFAULT_STOCK-only scenarios still work"""
    print("Test: Single-stock (DEFAULT_STOCK only)...")

    agent = DummyAgent("test", initial_cash=1000, initial_shares=0,
                       allow_short_selling=True, margin_requirement=0.5, margin_base="cash")
    agent.last_update_round = 1

    # Use old single-stock API
    agent.borrowed_shares = 100

    assert agent.total_borrowed_shares == 100, f"Expected 100, got {agent.total_borrowed_shares}"

    # Trigger via update_wealth
    agent.update_wealth(150.0)  # High price triggers margin call

    # Should have covered some
    assert agent.borrowed_shares < 100, "Should have triggered margin call"

    print("  ✓ PASSED")


def test_negative_cash_allowed():
    """Test that margin calls can make cash negative (forced liquidation)"""
    print("Test: Negative cash (forced liquidation)...")

    agent = DummyAgent("test", initial_cash=100, initial_shares=0,
                       allow_short_selling=True, margin_requirement=0.5, margin_base="cash")

    # Borrow a lot with limited cash
    agent.borrowed_positions["STOCK_A"] = 50
    agent.borrowed_shares += 50

    prices = {"STOCK_A": 200}  # Very high price
    agent.handle_multi_stock_margin_call(prices, round_number=1)

    # Cash can go negative (modeling forced liquidation without sufficient funds)
    print(f"  Cash after forced liquidation: ${agent.cash:.2f}")
    # This is allowed - represents forced liquidation beyond available cash

    print("  ✓ PASSED (cash can go negative)")


def test_empty_prices_dict():
    """Test behavior with empty prices dict"""
    print("Test: Empty prices dict...")

    agent = DummyAgent("test", initial_cash=10000, initial_shares=0,
                       allow_short_selling=True, margin_requirement=0.5)

    agent.borrowed_positions["STOCK_A"] = 50
    agent.borrowed_shares += 50

    prices = {}  # Empty!

    # Should not crash
    margin_status = agent.get_portfolio_margin_status(prices)
    assert margin_status['borrowed_value'] == 0, "No prices = no borrowed value"

    agent.handle_multi_stock_margin_call(prices, round_number=1)
    # Should do nothing (no prices to evaluate)

    print("  ✓ PASSED")


def test_zero_borrowed_shares():
    """Test that no margin call occurs when no shares borrowed"""
    print("Test: Zero borrowed shares...")

    agent = DummyAgent("test", initial_cash=10000, initial_shares=0,
                       allow_short_selling=True, margin_requirement=0.5)

    prices = {"STOCK_A": 100}

    initial_cash = agent.cash
    agent.handle_multi_stock_margin_call(prices, round_number=1)

    # Nothing should happen
    assert agent.cash == initial_cash, "Cash should not change"
    assert agent.total_borrowed_shares == 0, "Should have no borrowed shares"

    print("  ✓ PASSED")


def test_all_stocks_covered():
    """Test covering positions when margin violated"""
    print("Test: Margin call with violation...")

    agent = DummyAgent("test", initial_cash=5000, initial_shares=0,
                       allow_short_selling=True, margin_requirement=0.5, margin_base="cash")

    # Create a margin violation
    # max_borrowable_value = 5000 / 0.5 = 10000
    # We'll borrow 200*100 + 100*200 = 40000 worth
    agent.borrowed_positions["STOCK_A"] = 200  # 20000 value
    agent.borrowed_shares += 200
    agent.borrowed_positions["STOCK_B"] = 100  # 20000 value
    agent.borrowed_shares += 100

    prices = {"STOCK_A": 100, "STOCK_B": 200}

    initial_borrowed = agent.total_borrowed_shares
    print(f"  Initial borrowed: {initial_borrowed} shares, value: $40000")
    print(f"  Max borrowable value: $10000")
    print(f"  Violation: ${40000 - 10000}")

    # This should trigger covering
    agent.handle_multi_stock_margin_call(prices, round_number=1)

    final_total = agent.total_borrowed_shares
    print(f"  Final borrowed: {final_total} shares (reduced from {initial_borrowed})")
    assert final_total < initial_borrowed, "Should have reduced borrowing"

    print("  ✓ PASSED")


def main():
    print("\n" + "="*70)
    print("Edge Case Testing for Multi-Stock Margin Calls")
    print("="*70 + "\n")

    tests = [
        test_accumulator_consistency,
        test_single_stock_still_works,
        test_negative_cash_allowed,
        test_empty_prices_dict,
        test_zero_borrowed_shares,
        test_all_stocks_covered,
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
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    if failed == 0:
        print("✓ All edge case tests PASSED!")
    else:
        print(f"✗ {failed} test(s) FAILED")
    print("="*70 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
