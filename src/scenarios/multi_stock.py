"""
Multi-Stock Scenarios

Testing scenarios with multiple stocks:
- Pairs trading strategies
- Portfolio diversification
- Cross-stock arbitrage opportunities
"""

from .base import (
    SimulationScenario, DEFAULT_PARAMS,
    FUNDAMENTAL_WITH_DEFAULT_PARAMS, BASE_NUM_ROUNDS, BASE_INITIAL_CASH,
    BASE_INITIAL_SHARES, BASE_MAX_ORDER_SIZE, BASE_POSITION_LIMIT
)

SCENARIOS = {
    "multi_stock_pairs_trading": SimulationScenario(
        name="multi_stock_pairs_trading",
        description="Two tech stocks - test pairs trading with one overvalued, one undervalued",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 3,  # Short test run
            "IS_MULTI_STOCK": True,  # Flag to indicate multi-stock scenario
            "STOCKS": {
                "TECH_A": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 95.0,  # 5% overvalued
                    "REDEMPTION_VALUE": 95.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 4.75,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "TECH_B": {
                    "INITIAL_PRICE": 50.0,
                    "FUNDAMENTAL_PRICE": 55.0,  # 10% undervalued
                    "REDEMPTION_VALUE": 55.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.75,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH * 2,  # More cash for multi-stock
                'initial_positions': {  # NEW: Multi-stock positions
                    "TECH_A": 5000,
                    "TECH_B": 10000
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'multi_stock_test': 3,  # Use deterministic test agents
                    'hold_trader': 2  # And some hold traders
                }
            }
        }
    ),

    "multi_stock_llm_test": SimulationScenario(
        name="multi_stock_llm_test",
        description="Simple 2-round LLM test with 2 stocks and 4 value agents",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 2,  # Very short test
            "IS_MULTI_STOCK": True,
            "HIDE_FUNDAMENTAL_PRICE": False,  # Value agents need to see fundamental price
            "STOCKS": {
                "TECH_A": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 95.0,  # 5% overvalued
                    "REDEMPTION_VALUE": 95.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 4.75,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "TECH_B": {
                    "INITIAL_PRICE": 50.0,
                    "FUNDAMENTAL_PRICE": 55.0,  # 10% undervalued
                    "REDEMPTION_VALUE": 55.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.75,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
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
                    "TECH_A": 5000,
                    "TECH_B": 10000
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 4  # 4 LLM value investors
                }
            }
        }
    ),

    "multi_stock_llm_shorting_test": SimulationScenario(
        name="multi_stock_llm_shorting_test",
        description="Quick 2-round LLM test with short selling enabled - tests Issue #48 fix",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 2,  # Very short test
            "IS_MULTI_STOCK": True,
            "HIDE_FUNDAMENTAL_PRICE": False,
            "STOCKS": {
                "TECH_OVERVALUED": {
                    "INITIAL_PRICE": 110.0,
                    "FUNDAMENTAL_PRICE": 100.0,  # 10% overvalued - encourage shorting
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 3000,  # Per-stock lending pool
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.3,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "PHARMA_UNDERVALUED": {
                    "INITIAL_PRICE": 90.0,
                    "FUNDAMENTAL_PRICE": 100.0,  # 11% undervalued - encourage buying
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 5000,  # Different pool size
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.5,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.3,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': True,  # ENABLE SHORT SELLING for LLMs
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH * 3,  # Extra cash for margin
                'initial_positions': {
                    "TECH_OVERVALUED": 1000,
                    "PHARMA_UNDERVALUED": 1000
                },
                'max_order_size': 200,
                'agent_composition': {
                    'value': 2  # 2 LLM value investors - should short TECH, buy PHARMA
                },
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1,
                    'allow_partial_borrows': True
                },
                'interest_model': {
                    'rate': 0.01,
                    'compound_frequency': 1
                }
            }
        }
    ),

    "multi_stock_trade_test": SimulationScenario(
        name="multi_stock_trade_test",
        description="Multi-stock test with buy/sell agents that WILL execute trades",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 3,  # Short test
            "IS_MULTI_STOCK": True,
            "HIDE_FUNDAMENTAL_PRICE": False,
            "STOCKS": {
                "TECH_A": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 5.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "TECH_B": {
                    "INITIAL_PRICE": 50.0,
                    "FUNDAMENTAL_PRICE": 50.0,
                    "REDEMPTION_VALUE": 50.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.5,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH * 3,  # Plenty of cash for buying
                'initial_positions': {
                    "TECH_A": 1000,  # Some shares for sellers
                    "TECH_B": 1000   # Some shares for sellers
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'multi_stock_buy': 2,   # 2 agents that always buy
                    'multi_stock_sell': 2   # 2 agents that always sell
                }
            }
        }
    ),

    "multi_stock_cash_bug_test": SimulationScenario(
        name="multi_stock_cash_bug_test",
        description="Test cash over-commitment bug fix: 3 stocks with limited agent cash",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 3,  # Short test
            "IS_MULTI_STOCK": True,
            "HIDE_FUNDAMENTAL_PRICE": False,
            "STOCKS": {
                "STOCK_A": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 105.0,  # Undervalued to trigger buying
                    "REDEMPTION_VALUE": 105.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 5.25,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "STOCK_B": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 105.0,  # Undervalued to trigger buying
                    "REDEMPTION_VALUE": 105.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 5.25,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "STOCK_C": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 105.0,  # Undervalued to trigger buying
                    "REDEMPTION_VALUE": 105.0,
                    "TRANSACTION_COST": 0.0,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 5.25,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': 12000.0,  # LIMITED CASH: Can't afford all 3 stocks
                'initial_positions': {
                    "STOCK_A": 50,  # Small positions to avoid triggering sell logic
                    "STOCK_B": 50,
                    "STOCK_C": 50
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'multi_stock_buy': 2,        # Will try to buy 50 shares × $101 × 3 = $15,150
                    'multi_stock_value': 2,      # Will try to buy 100 shares × $100 × 3 = $30,000
                }
            }
        }
    ),

    "multi_stock_short_selling_test": SimulationScenario(
        name="multi_stock_short_selling_test",
        description="Multi-stock short selling with per-stock borrowing pools - demonstrates Issue #48 fix",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "IS_MULTI_STOCK": True,
            "HIDE_FUNDAMENTAL_PRICE": False,
            "STOCKS": {
                "TECH_OVERVALUED": {
                    "INITIAL_PRICE": 120.0,
                    "FUNDAMENTAL_PRICE": 100.0,  # 20% overvalued - encourage shorting
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 5000,  # Limited pool for TECH_OVERVALUED
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.3,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "PHARMA_OVERVALUED": {
                    "INITIAL_PRICE": 80.0,
                    "FUNDAMENTAL_PRICE": 60.0,  # 33% overvalued - encourage shorting
                    "REDEMPTION_VALUE": 60.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 10000,  # Larger pool for PHARMA_OVERVALUED
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 1.5,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.3,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "ENERGY_FAIR": {
                    "INITIAL_PRICE": 50.0,
                    "FUNDAMENTAL_PRICE": 50.0,  # Fairly valued - minimal shorting
                    "REDEMPTION_VALUE": 50.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 2000,  # Small pool for ENERGY_FAIR
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 1.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.3,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': True,  # ENABLE SHORT SELLING
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH * 2,
                'initial_positions': {
                    "TECH_OVERVALUED": 1000,
                    "PHARMA_OVERVALUED": 2000,
                    "ENERGY_FAIR": 1000
                },
                'max_order_size': 500,  # Smaller orders to test borrowing dynamics
                'agent_composition': {
                    'multi_stock_value': 3,  # Value traders will short overvalued stocks
                    'multi_stock_sell': 2  # Sell agents will short when they run out of shares
                },
                'borrow_model': {
                    'rate': 0.02,  # 2% borrow fee per round
                    'payment_frequency': 1,
                    'allow_partial_borrows': True  # Allow partial fills
                },
                'interest_model': {
                    'rate': 0.01,
                    'compound_frequency': 1
                }
            }
        }
    ),

    "gptoss_multistock_memory_test": SimulationScenario(
        name="gptoss_multistock_memory_test",
        description="Multi-stock memory test with GPT-OSS - 3 stocks with different mispricing levels",
        parameters={
            **DEFAULT_PARAMS,
            "MODEL_OPEN_AI": "gpt-oss-120b",  # Use GPT-OSS reasoning model
            "NUM_ROUNDS": 8,  # Medium length for memory testing
            "IS_MULTI_STOCK": True,
            "HIDE_FUNDAMENTAL_PRICE": False,
            "STOCKS": {
                "TECH_OVERVALUED": {
                    "INITIAL_PRICE": 110.0,
                    "FUNDAMENTAL_PRICE": 100.0,  # 10% overvalued
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 5000,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "PHARMA_UNDERVALUED": {
                    "INITIAL_PRICE": 85.0,
                    "FUNDAMENTAL_PRICE": 100.0,  # 15% undervalued
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 5000,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.5,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "ENERGY_FAIR": {
                    "INITIAL_PRICE": 50.0,
                    "FUNDAMENTAL_PRICE": 50.0,  # Fairly valued
                    "REDEMPTION_VALUE": 50.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 3000,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 1.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # BOTH features enabled for memory testing
                'MEMORY_ENABLED': True,
                'SOCIAL_ENABLED': True,

                'allow_short_selling': True,  # Enable shorting for overvalued stock
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH * 3,  # Extra cash for multi-stock
                'initial_positions': {
                    "TECH_OVERVALUED": 1000,
                    "PHARMA_UNDERVALUED": 1000,
                    "ENERGY_FAIR": 1000
                },
                'max_order_size': 500,
                'agent_composition': {
                    'influencer': 1,
                    'herd_follower': 2,
                    'value': 2,
                    'contrarian': 1,
                },
                'borrow_model': {
                    'rate': 0.02,
                    'payment_frequency': 1,
                    'allow_partial_borrows': True
                },
                'interest_model': {
                    'rate': 0.01,
                    'compound_frequency': 1
                }
            }
        }
    ),
}
