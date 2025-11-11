"""
Short Selling Scenarios

Testing scenarios focused on short selling mechanics:
- Basic short selling with borrow costs
- Aggressive short sellers with overpriced stocks
- Partial vs all-or-nothing borrow fills
"""

from .base import (
    SimulationScenario, DEFAULT_PARAMS,
    FUNDAMENTAL_WITH_DEFAULT_PARAMS, BASE_NUM_ROUNDS, BASE_INITIAL_CASH,
    BASE_INITIAL_SHARES, BASE_MAX_ORDER_SIZE, BASE_POSITION_LIMIT
)

SCENARIOS = {
    "short_selling_benchmark": SimulationScenario(
        name="short_selling_benchmark",
        description="Benchmark scenario with short selling enabled and borrow costs",
        parameters={
            **DEFAULT_PARAMS,
            "LENDABLE_SHARES": 20000,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
            }
        }
    ),
    "test_short_seller": SimulationScenario(
        name="test_short_seller",
        description="Test scenario with new short_seller agent and overvalued price",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "INITIAL_PRICE": 35.0,  # 25% above fundamental to encourage shorting
            "LENDABLE_SHARES": 30000,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'agent_composition': {
                    'short_seller': 2,      # New aggressive short sellers
                    'optimistic': 1,        # Provides buy-side liquidity
                    'market_maker': 1       # Provides liquidity
                },
                'type_specific_params': {
                    'short_seller': {
                        'initial_cash': 1.5 * BASE_INITIAL_CASH,
                        'initial_shares': int(0.5 * BASE_INITIAL_SHARES)  # Only 5000 shares
                    },
                    'optimistic': {
                        'initial_cash': 2.0 * BASE_INITIAL_CASH,
                        'initial_shares': int(0.2 * BASE_INITIAL_SHARES)
                    },
                    'market_maker': {
                        'initial_cash': 3.0 * BASE_INITIAL_CASH,
                        'initial_shares': int(2.0 * BASE_INITIAL_SHARES)
                    }
                }
            }
        }
    ),
    "aggressive_short_selling": SimulationScenario(
        name="aggressive_short_selling",
        description="Force aggressive short selling with extreme overpricing and low dividends",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 45.0,  # 60% above fundamental ($28)!
            "LENDABLE_SHARES": 50000,  # Plenty of shares to borrow
            "DIVIDEND_PARAMS": {
                'type': 'stochastic',
                'base_dividend': 0.2,  # Very low dividend (was 1.4)
                'dividend_frequency': 1,
                'dividend_growth': 0.0,
                'dividend_probability': 0.5,
                'dividend_variation': 0.1,  # Very low variation
                'destination': 'dividend'
            },
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.01,  # Lower borrow cost (1% vs 2%)
                    'payment_frequency': 1
                },
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'agent_composition': {
                    'short_seller': 3,      # 3 aggressive short sellers
                    'optimistic': 2,        # 2 optimistic buyers for liquidity
                },
                'type_specific_params': {
                    'short_seller': {
                        'initial_cash': 3.0 * BASE_INITIAL_CASH,  # Lots of cash for margin
                        'initial_shares': 0  # ZERO shares - MUST borrow to short!
                    },
                    'optimistic': {
                        'initial_cash': 3.0 * BASE_INITIAL_CASH,  # Lots of cash to buy
                        'initial_shares': int(0.1 * BASE_INITIAL_SHARES)  # Very few shares
                    }
                }
            }
        }
    ),
    "partial_borrow_test_disabled": SimulationScenario(
        name="partial_borrow_test_disabled",
        description="Test partial borrow fills feature - DISABLED (all-or-nothing behavior)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 28.0,
            "LENDABLE_SHARES": 1000,  # Limited lending pool - only 1000 shares available
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.01,
                    'payment_frequency': 1,
                    'allow_partial_borrows': False  # DISABLED - all-or-nothing
                },
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'short_sell_trader': 3,  # 3 short sellers, each trying to short 500 shares = 1500 total
                    'buy_trader': 2,         # 2 buyers for liquidity
                },
                'type_specific_params': {
                    'short_sell_trader': {
                        'initial_cash': 3.0 * BASE_INITIAL_CASH,  # Lots of cash for margin
                        'initial_shares': 0  # ZERO shares - MUST borrow to short!
                    },
                    'buy_trader': {
                        'initial_cash': 5.0 * BASE_INITIAL_CASH,
                        'initial_shares': 0
                    }
                }
            }
        }
    ),
    "partial_borrow_test_enabled": SimulationScenario(
        name="partial_borrow_test_enabled",
        description="Test partial borrow fills feature - ENABLED (partial fills allowed)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 28.0,
            "LENDABLE_SHARES": 1000,  # Limited lending pool - only 1000 shares available
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.01,
                    'payment_frequency': 1,
                    'allow_partial_borrows': True  # ENABLED - allows partial fills
                },
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'short_sell_trader': 3,  # 3 short sellers, each trying to short 500 shares = 1500 total
                    'buy_trader': 2,         # 2 buyers for liquidity
                },
                'type_specific_params': {
                    'short_sell_trader': {
                        'initial_cash': 3.0 * BASE_INITIAL_CASH,  # Lots of cash for margin
                        'initial_shares': 0  # ZERO shares - MUST borrow to short!
                    },
                    'buy_trader': {
                        'initial_cash': 5.0 * BASE_INITIAL_CASH,
                        'initial_shares': 0
                    }
                }
            }
        }
    ),
}
