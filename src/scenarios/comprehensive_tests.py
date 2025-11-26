"""
Comprehensive Test Scenarios

Systematic test scenarios covering all combinations of:
- Single-stock vs Multi-stock
- Leverage vs No leverage
- Short selling vs No short selling

This ensures all feature combinations work correctly.
"""

from .base import (
    SimulationScenario, DEFAULT_PARAMS,
    BASE_NUM_ROUNDS, BASE_INITIAL_CASH,
    BASE_INITIAL_SHARES, BASE_MAX_ORDER_SIZE, BASE_POSITION_LIMIT
)

SCENARIOS = {
    # ==================== SINGLE-STOCK SCENARIOS ====================

    "single_basic": SimulationScenario(
        name="single_basic",
        description="Single-stock: Basic trading (no leverage, no short selling)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "INITIAL_PRICE": 100.0,
            "FUNDAMENTAL_PRICE": 100.0,
            "REDEMPTION_VALUE": 100.0,
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'optimistic': 2,
                    'pessimistic': 2
                }
            }
        }
    ),

    "single_short": SimulationScenario(
        name="single_short",
        description="Single-stock: Short selling enabled (no leverage)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "INITIAL_PRICE": 120.0,  # Overvalued to encourage shorting
            "FUNDAMENTAL_PRICE": 100.0,
            "REDEMPTION_VALUE": 100.0,
            "LENDABLE_SHARES": 20000,  # Share lending pool for short selling
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'short_seller': 2,  # LLM short sellers
                    'buy_trader': 2  # Deterministic buyers as counterparty
                },
                'type_specific_params': {
                    'short_seller': {
                        'initial_shares': 0,  # No shares - MUST borrow to short
                        'initial_cash': 500000  # Extra cash for margin
                    }
                }
            }
        }
    ),

    "single_leverage": SimulationScenario(
        name="single_leverage",
        description="Single-stock: Leverage enabled (no short selling)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "INITIAL_PRICE": 80.0,  # Undervalued to encourage leveraged buying
            "FUNDAMENTAL_PRICE": 100.0,
            "REDEMPTION_VALUE": 100.0,
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT * 2,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'margin_buyer': 2,    # Deterministic margin buyers that use leverage
                    'sell_trader': 2    # Deterministic sellers as counterparty
                },
                'type_specific_params': {
                    'margin_buyer': {
                        'initial_cash': 100000,  # Some cash, but will borrow more
                        'initial_shares': 10000  # Equity for borrowing power
                    },
                    'sell_trader': {
                        'initial_cash': 500000,
                        'initial_shares': 20000  # Lots of shares to sell
                    }
                },
                'leverage_params': {
                    'enabled': True,
                    'max_leverage_ratio': 2.0,
                    'initial_margin': 0.5,
                    'maintenance_margin': 0.25,
                    'interest_rate': 0.05,
                    'cash_lending_pool': float('inf'),
                    'allow_partial_borrows': True,
                }
            }
        }
    ),

    "single_leverage_short": SimulationScenario(
        name="single_leverage_short",
        description="Single-stock: Leverage + Short selling (all features)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "INITIAL_PRICE": 100.0,
            "FUNDAMENTAL_PRICE": 100.0,
            "REDEMPTION_VALUE": 100.0,
            "LENDABLE_SHARES": 20000,  # Share lending pool for short selling
            "ENABLE_INTRA_ROUND_MARGIN_CHECKING": True,  # Test margin detection during matching
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT * 2,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'margin_buyer': 1,      # Deterministic: uses leverage (borrowed_cash)
                    'short_seller': 1,      # Deterministic: uses short selling (borrowed_shares)
                    'buy_trader': 1,        # Counterparty for short sellers
                    'sell_trader': 1        # Counterparty for margin buyers
                },
                'type_specific_params': {
                    'margin_buyer': {
                        'initial_cash': 100000,  # Some cash, will borrow more
                        'initial_shares': 10000  # Equity for borrowing power
                    },
                    'short_seller': {
                        'initial_shares': 0,     # No shares - must borrow to short
                        'initial_cash': 500000   # Cash for margin
                    },
                    'buy_trader': {
                        'initial_cash': 500000,
                        'initial_shares': 5000
                    },
                    'sell_trader': {
                        'initial_cash': 500000,
                        'initial_shares': 20000  # Lots of shares to sell
                    }
                },
                'leverage_params': {
                    'enabled': True,
                    'max_leverage_ratio': 2.0,
                    'initial_margin': 0.5,
                    'maintenance_margin': 0.25,
                    'interest_rate': 0.05,
                    'cash_lending_pool': float('inf'),
                    'allow_partial_borrows': True,
                }
            }
        }
    ),

    "margin_violation_test": SimulationScenario(
        name="margin_violation_test",
        description="Test scenario: Deterministic agents to force margin violations",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 8,
            "INITIAL_PRICE": 50.0,  # Low starting price
            "FUNDAMENTAL_PRICE": 100.0,  # High fundamental - buyers push toward this
            "REDEMPTION_VALUE": 100.0,
            "FUNDAMENTAL_VOLATILITY": 0.0,  # No noise - clean test
            "LENDABLE_SHARES": 10000,  # Plenty of shares for shorting
            "ENABLE_INTRA_ROUND_MARGIN_CHECKING": True,  # Enable margin checking
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'margin_requirement': 3.0,  # ULTRA-TIGHT margin (300% collateral!) - violates easily even after sale
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT * 2,
                'initial_cash': 150000,  # Enough to borrow 500 shares at 3.0 margin, violates as price rises!
                'initial_shares': 0,  # Start with NO shares - forces short seller to borrow!
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'short_sell_trader': 1,  # Deterministic short seller - MUST borrow
                    'buy_trader': 2,         # Deterministic buyers - push price UP
                    'hold_trader': 1         # Provides liquidity (buys from short seller)
                },
                'leverage_params': {
                    'enabled': True,
                    'max_leverage_ratio': 2.0,
                    'initial_margin': 0.5,
                    'maintenance_margin': 0.25,
                    'interest_rate': 0.05,
                    'cash_lending_pool': float('inf'),
                    'allow_partial_borrows': True,
                }
            }
        }
    ),

    "short_squeeze_test": SimulationScenario(
        name="short_squeeze_test",
        description="Test scenario: Short squeeze with timed aggressive buyer to trigger margin calls",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 8,  # Extended to see margin calls execute after price spike
            "INITIAL_PRICE": 50.0,
            "FUNDAMENTAL_PRICE": 100.0,
            "REDEMPTION_VALUE": 100.0,
            "FUNDAMENTAL_VOLATILITY": 0.0,  # No noise - clean test
            "LENDABLE_SHARES": 10000,  # Plenty of shares for shorting
            "ENABLE_INTRA_ROUND_MARGIN_CHECKING": True,  # Enable margin checking
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'margin_requirement': 1.5,  # 150% margin requirement (more realistic)
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT * 2,
                'initial_cash': 38000,  # Enough to borrow 500 shares @ $50 with 150% margin
                'initial_shares': 100,  # Market makers need shares to sell; short seller still borrows 400 shares (sells 500 > has 100)
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'short_sell_trader': 1,         # Builds 500 share short position
                    'squeeze_buyer': 1,             # ACTIVATES round 3: massive squeeze!
                    'deterministic_market_maker': 2, # Provides liquidity with buy and sell orders
                    'hold_trader': 1                # Provides initial liquidity
                },
                'leverage_params': {
                    'enabled': True,
                    'max_leverage_ratio': 2.0,
                    'initial_margin': 0.5,
                    'maintenance_margin': 0.25,
                    'interest_rate': 0.05,
                    'cash_lending_pool': float('inf'),
                    'allow_partial_borrows': True,
                }
            }
        }
    ),

    # ==================== MULTI-STOCK SCENARIOS ====================

    "multi_basic": SimulationScenario(
        name="multi_basic",
        description="Multi-stock: Basic trading (no leverage, no short selling)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "IS_MULTI_STOCK": True,
            "STOCKS": {
                "STOCK_A": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 5.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "STOCK_B": {
                    "INITIAL_PRICE": 50.0,
                    "FUNDAMENTAL_PRICE": 50.0,
                    "REDEMPTION_VALUE": 50.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 2.5,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH * 2,
                'initial_positions': {
                    "STOCK_A": 1000,
                    "STOCK_B": 1000
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'multi_stock_buy': 2,
                    'multi_stock_sell': 2
                }
            }
        }
    ),

    "multi_short": SimulationScenario(
        name="multi_short",
        description="Multi-stock: Short selling enabled (no leverage)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "IS_MULTI_STOCK": True,
            "STOCKS": {
                "TECH_OVERVALUED": {
                    "INITIAL_PRICE": 110.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 5000,  # Share lending pool
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 5.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "PHARMA_UNDERVALUED": {
                    "INITIAL_PRICE": 85.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 5000,  # Share lending pool
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 5.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH * 2,
                'initial_positions': {
                    "TECH_OVERVALUED": 0,
                    "PHARMA_UNDERVALUED": 0
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'multi_stock_short_seller': 2,  # Deterministic short sellers
                    'multi_stock_squeeze_buyer': 2  # Buyers as counterparty
                },
                'type_specific_params': {
                    'multi_stock_short_seller': {
                        'initial_positions': {
                            "TECH_OVERVALUED": 0,  # 0 shares - must borrow to short
                            "PHARMA_UNDERVALUED": 0
                        },
                        'initial_cash': 1000000   # Cash for margin
                    },
                    'multi_stock_squeeze_buyer': {
                        'initial_positions': {
                            "TECH_OVERVALUED": 0,  # Will buy from short sellers
                            "PHARMA_UNDERVALUED": 0
                        },
                        'initial_cash': 2000000  # Cash to buy shares
                    }
                }
            }
        }
    ),

    "multi_leverage": SimulationScenario(
        name="multi_leverage",
        description="Multi-stock: Leverage enabled (no short selling)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "IS_MULTI_STOCK": True,
            "STOCKS": {
                "STOCK_A": {
                    "INITIAL_PRICE": 80.0,  # Undervalued
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 5.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "STOCK_B": {
                    "INITIAL_PRICE": 40.0,  # Undervalued
                    "FUNDAMENTAL_PRICE": 50.0,
                    "REDEMPTION_VALUE": 50.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 2.5,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT * 2,
                'initial_cash': BASE_INITIAL_CASH * 2,
                'initial_positions': {
                    "STOCK_A": 1000,
                    "STOCK_B": 1000
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'leverage_trader': 2,
                    'default': 2
                },
                'leverage_params': {
                    'enabled': True,
                    'max_leverage_ratio': 2.0,
                    'initial_margin': 0.5,
                    'maintenance_margin': 0.25,
                    'interest_rate': 0.05,
                    'cash_lending_pool': float('inf'),
                    'allow_partial_borrows': True,
                }
            }
        }
    ),

    "multi_leverage_short": SimulationScenario(
        name="multi_leverage_short",
        description="Multi-stock: Leverage + Short selling (all features)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "IS_MULTI_STOCK": True,
            "STOCKS": {
                "TECH_OVERVALUED": {
                    "INITIAL_PRICE": 110.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 5000,  # Share lending pool
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 5.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "PHARMA_UNDERVALUED": {
                    "INITIAL_PRICE": 85.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 5000,  # Share lending pool
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 5.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT * 2,
                'initial_cash': BASE_INITIAL_CASH * 2,
                'initial_positions': {
                    "TECH_OVERVALUED": 0,
                    "PHARMA_UNDERVALUED": 0
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'leverage_trader': 2,          # LLM agents that use leverage (borrowed_cash)
                    'multi_stock_short_seller': 2  # Deterministic short sellers (borrowed_shares)
                },
                'type_specific_params': {
                    'leverage_trader': {
                        'initial_positions': {
                            "TECH_OVERVALUED": 0,
                            "PHARMA_UNDERVALUED": 0
                        },
                        'initial_cash': 2000000  # Lots of cash, will borrow more
                    },
                    'multi_stock_short_seller': {
                        'initial_positions': {
                            "TECH_OVERVALUED": 0,  # No shares - must borrow to short
                            "PHARMA_UNDERVALUED": 0
                        },
                        'initial_cash': 1000000  # Cash for margin
                    }
                },
                'leverage_params': {
                    'enabled': True,
                    'max_leverage_ratio': 2.0,
                    'initial_margin': 0.5,
                    'maintenance_margin': 0.25,
                    'interest_rate': 0.05,
                    'cash_lending_pool': float('inf'),
                    'allow_partial_borrows': True,
                }
            }
        }
    ),

    # ==================== ADDITIONAL TEST: ACTIVE SHORT SELLING ====================

    "multi_short_active": SimulationScenario(
        name="multi_short_active",
        description="Multi-stock: Active short selling with short_sell_trader agents",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "IS_MULTI_STOCK": True,
            "STOCKS": {
                "TECH_OVERVALUED": {
                    "INITIAL_PRICE": 120.0,  # Significantly overvalued to trigger shorting
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 10000,  # Large lending pool
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 5.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "PHARMA_OVERVALUED": {
                    "INITIAL_PRICE": 150.0,  # Also overvalued
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 10000,  # Large lending pool
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 5.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT * 2,
                'initial_cash': BASE_INITIAL_CASH * 2,
                'initial_positions': {
                    "TECH_OVERVALUED": 1000,  # Initial shares for liquidity
                    "PHARMA_OVERVALUED": 1000
                },
                'max_order_size': BASE_MAX_ORDER_SIZE * 2,
                'agent_composition': {
                    'multi_stock_value': 3,  # Value traders will short overvalued stocks
                    'multi_stock_buy': 2  # Buy-side liquidity
                },
                'type_specific_params': {
                    'multi_stock_value': {
                        'initial_cash': 3.0 * BASE_INITIAL_CASH,
                        'initial_shares': 500  # Some shares, will sell/short more
                    },
                    'multi_stock_buy': {
                        'initial_cash': 5.0 * BASE_INITIAL_CASH,
                        'initial_shares': 2000  # Lots of shares to sell for liquidity
                    }
                }
            }
        }
    ),

    # ==================== STRESS TEST: LEVERAGE MARGIN CALLS ====================

    "leverage_stress_test": SimulationScenario(
        name="leverage_stress_test",
        description="Stress test: Force leverage usage and margin calls with deterministic agents (fast)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 50.0,   # LOW price - undervalued
            # Fundamental = expected_dividend / interest_rate = 5 / 0.05 = 100
            "DIVIDEND_PARAMS": {
                'type': 'fixed',
                'base_dividend': 5.0,
                'dividend_frequency': 1,
                'dividend_growth': 0.0,
                'dividend_probability': 1.0,  # Deterministic dividends
                'dividend_variation': 0.0,
                'destination': 'dividend'
            },
            "ENABLE_INTRA_ROUND_MARGIN_CHECKING": True,
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': 50000,
                'initial_cash': 50000,    # Very LOW cash to force leverage
                'initial_shares': 5000,
                'max_order_size': 2000,   # High order size
                'agent_composition': {
                    'gap_trader': 2,      # Buyers who will use leverage
                    'sell_trader': 2,     # Deterministic sellers to provide liquidity
                },
                'type_specific_params': {
                    'sell_trader': {
                        'initial_cash': 500000,   # Lots of cash
                        'initial_shares': 20000,  # Lots of shares to sell
                    }
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,       # Buy when price < 95% of fundamental
                        'max_proportion': 0.8,   # Aggressive - use 80% of buying power
                        'scaling_factor': 3.0    # Scale up orders
                    }
                },
                'leverage_params': {
                    'enabled': True,
                    'max_leverage_ratio': 3.0,
                    'initial_margin': 0.5,
                    'maintenance_margin': 0.25,
                    'interest_rate': 0.05,
                    'cash_lending_pool': float('inf'),
                    'allow_partial_borrows': True,
                }
            }
        }
    ),

    # ==================== DIVERSE LLM SCENARIO ====================

    "llm_long_short": SimulationScenario(
        name="llm_long_short",
        description="Multi-stock long-short: LLMs can go long one stock, short another",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "IS_MULTI_STOCK": True,
            "STOCKS": {
                "UNDERVALUED": {
                    "INITIAL_PRICE": 80.0,       # Trading below fundamental
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 20000,
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 5.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "OVERVALUED": {
                    "INITIAL_PRICE": 120.0,      # Trading above fundamental
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 20000,
                    "DIVIDEND_PARAMS": {
                        'type': 'fixed',
                        'base_dividend': 5.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 1.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT * 2,
                'initial_cash': BASE_INITIAL_CASH * 2,
                'initial_positions': {
                    "UNDERVALUED": 0,
                    "OVERVALUED": 0
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'long_short': 2,    # Specialized pairs trader: long UNDERVALUED, short OVERVALUED
                    'speculator': 1,    # Should identify mispricing
                    'optimistic': 1     # Counterparty: buys
                },
                'leverage_params': {
                    'enabled': True,
                    'max_leverage_ratio': 2.0,
                    'initial_margin': 0.5,
                    'maintenance_margin': 0.25,
                    'interest_rate': 0.05,
                    'cash_lending_pool': float('inf'),
                    'allow_partial_borrows': True,
                }
            }
        }
    ),

    "llm_diverse": SimulationScenario(
        name="llm_diverse",
        description="Diverse LLM agents: All features enabled with variety of agent personalities",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "INITIAL_PRICE": 100.0,
            "FUNDAMENTAL_PRICE": 100.0,
            "REDEMPTION_VALUE": 100.0,
            "LENDABLE_SHARES": 50000,  # Share lending pool for short selling
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT * 2,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'leverage_trader': 1,   # Aggressive leverage user
                    'short_seller': 1,      # Short selling specialist
                    'optimistic': 1,        # Bullish buyer
                    'pessimistic': 1,       # Bearish seller
                    'speculator': 1,        # Market inefficiency hunter
                    'default': 1            # Neutral baseline
                },
                'type_specific_params': {
                    'short_seller': {
                        'initial_shares': 0,    # No shares - MUST borrow to short
                        'initial_cash': 500000  # Extra cash for margin
                    }
                },
                'leverage_params': {
                    'enabled': True,
                    'max_leverage_ratio': 2.0,
                    'initial_margin': 0.5,
                    'maintenance_margin': 0.25,
                    'interest_rate': 0.05,
                    'cash_lending_pool': float('inf'),
                    'allow_partial_borrows': True,
                }
            }
        }
    ),
}
