#!/usr/bin/env python3
"""Test defensive handling of DEFAULT_STOCK in prices dict"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

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


def test_default_stock_in_prices_margin_status():
    """Test that DEFAULT_STOCK in prices dict is safely ignored in margin status calculation"""
    print("Test: DEFAULT_STOCK in prices (margin status)...")

    agent = DummyAgent("test", initial_cash=10000, initial_shares=0,
                       allow_short_selling=True, margin_requirement=0.5, margin_base="wealth")

    # Set up borrowed positions with accumulator pattern
    agent.borrowed_positions["STOCK_A"] = 50
    agent.borrowed_shares += 50
    agent.borrowed_positions["STOCK_B"] = 30
    agent.borrowed_shares += 30

    # Prices dict INCORRECTLY includes DEFAULT_STOCK (defensive test)
    prices_with_default = {
        "STOCK_A": 100,
        "STOCK_B": 200,
        "DEFAULT_STOCK": 150  # Should be ignored!
    }

    # Calculate margin status - should ignore DEFAULT_STOCK
    status = agent.get_portfolio_margin_status(prices_with_default)

    # Expected borrowed value: 50*100 + 30*200 = 5000 + 6000 = 11000
    # Should NOT include DEFAULT_STOCK's 80 * 150 = 12000
    expected_borrowed_value = 11000
    assert status['borrowed_value'] == expected_borrowed_value, \
        f"Expected {expected_borrowed_value}, got {status['borrowed_value']}"

    # Net position value should also ignore DEFAULT_STOCK
    # Expected: (0-50)*100 + (0-30)*200 = -5000 - 6000 = -11000
    expected_net = -11000
    assert status['net_position_value'] == expected_net, \
        f"Expected {expected_net}, got {status['net_position_value']}"

    print("  ✓ PASSED - DEFAULT_STOCK correctly ignored in calculations")


def test_default_stock_in_prices_margin_call():
    """Test that DEFAULT_STOCK in prices dict doesn't cause issues in margin call"""
    print("Test: DEFAULT_STOCK in prices (margin call)...")

    agent = DummyAgent("test", initial_cash=5000, initial_shares=0,
                       allow_short_selling=True, margin_requirement=0.5, margin_base="cash")

    # Create a margin violation
    agent.borrowed_positions["STOCK_A"] = 100  # Value: 10000
    agent.borrowed_shares += 100
    agent.borrowed_positions["STOCK_B"] = 50   # Value: 10000
    agent.borrowed_shares += 50

    # Prices dict INCORRECTLY includes DEFAULT_STOCK
    prices_with_default = {
        "STOCK_A": 100,
        "STOCK_B": 200,
        "DEFAULT_STOCK": 150  # Should be ignored!
    }

    initial_default_stock = agent.borrowed_shares
    initial_stock_a = agent.borrowed_positions["STOCK_A"]
    initial_stock_b = agent.borrowed_positions["STOCK_B"]

    # Trigger margin call
    agent.handle_multi_stock_margin_call(prices_with_default, round_number=1)

    # DEFAULT_STOCK should still be consistent with sum of other stocks
    final_stock_a = agent.borrowed_positions.get("STOCK_A", 0)
    final_stock_b = agent.borrowed_positions.get("STOCK_B", 0)
    final_default = agent.borrowed_shares

    # After covering, DEFAULT_STOCK should equal sum of others
    assert abs(final_default - (final_stock_a + final_stock_b)) < 0.01, \
        f"Accumulator inconsistent: DEFAULT={final_default}, A+B={final_stock_a + final_stock_b}"

    # Some shares should have been covered
    assert agent.total_borrowed_shares < 150, "Should have covered some shares"

    # DEFAULT_STOCK in borrowed_positions should not have been directly modified
    # (it's only modified via the accumulator pattern)
    assert "DEFAULT_STOCK" not in agent.positions or agent.positions.get("DEFAULT_STOCK", 0) == 0, \
        "DEFAULT_STOCK should not have been added to positions"

    print("  ✓ PASSED - DEFAULT_STOCK safely ignored in margin call")


def test_only_default_stock_in_prices():
    """Test edge case where only DEFAULT_STOCK is in prices (should handle gracefully)"""
    print("Test: Only DEFAULT_STOCK in prices...")

    agent = DummyAgent("test", initial_cash=10000, initial_shares=0,
                       allow_short_selling=True, margin_requirement=0.5, margin_base="cash")

    # Has real borrowed positions
    agent.borrowed_positions["STOCK_A"] = 50
    agent.borrowed_shares += 50

    # But prices only has DEFAULT_STOCK (weird but should handle)
    prices_only_default = {"DEFAULT_STOCK": 100}

    # Should handle gracefully - no crashes
    status = agent.get_portfolio_margin_status(prices_only_default)

    # With no real stocks in prices, borrowed value should be 0
    assert status['borrowed_value'] == 0, \
        f"Expected 0 borrowed value, got {status['borrowed_value']}"

    # No margin call should be triggered (no prices to evaluate)
    initial_borrowed = agent.total_borrowed_shares
    agent.handle_multi_stock_margin_call(prices_only_default, round_number=1)

    # Borrowed shares should be unchanged (no action possible without prices)
    assert agent.total_borrowed_shares == initial_borrowed, \
        "Should not modify positions when no valid prices available"

    print("  ✓ PASSED - Handles DEFAULT_STOCK-only prices gracefully")


def test_default_stock_in_update_wealth():
    """Test that DEFAULT_STOCK in prices dict is safely ignored in wealth calculation"""
    print("Test: DEFAULT_STOCK in prices (update_wealth)...")

    agent = DummyAgent("test", initial_cash=10000, initial_shares=0,
                       allow_short_selling=True, margin_requirement=0.5, margin_base="wealth")

    # Set up borrowed positions with accumulator pattern
    agent.borrowed_positions["STOCK_A"] = 80
    agent.borrowed_shares += 80
    agent.borrowed_positions["STOCK_B"] = 30
    agent.borrowed_shares += 30

    # Prices dict INCORRECTLY includes DEFAULT_STOCK (defensive test)
    prices_with_default = {
        "STOCK_A": 100,
        "STOCK_B": 200,
        "DEFAULT_STOCK": 150  # Should be ignored!
    }

    # Update wealth - should ignore DEFAULT_STOCK
    agent.update_wealth(prices_with_default)

    # Expected share value: (0-80)*100 + (0-30)*200 = -8000 - 6000 = -14000
    # Should NOT include DEFAULT_STOCK's (0-110)*150 = -16500
    expected_share_value = -14000
    expected_wealth = 10000 + expected_share_value  # 10000 - 14000 = -4000

    assert abs(agent.wealth - expected_wealth) < 0.01, \
        f"Expected wealth {expected_wealth}, got {agent.wealth}"

    print("  ✓ PASSED - DEFAULT_STOCK correctly ignored in wealth calculation")


def main():
    print("\n" + "="*70)
    print("Defensive DEFAULT_STOCK Handling Tests")
    print("="*70 + "\n")

    tests = [
        test_default_stock_in_prices_margin_status,
        test_default_stock_in_prices_margin_call,
        test_only_default_stock_in_prices,
        test_default_stock_in_update_wealth,
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
        print("✓ All defensive tests PASSED!")
    else:
        print(f"✗ {failed} test(s) FAILED")
    print("="*70 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
