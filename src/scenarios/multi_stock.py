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

    "multi_stock_margin_call_test": SimulationScenario(
        name="multi_stock_margin_call_test",
        description="Multi-stock margin call test - short squeeze triggers margin calls on multiple stocks",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 8,  # Extended to see margin calls execute after price spike
            "IS_MULTI_STOCK": True,
            "HIDE_FUNDAMENTAL_PRICE": False,
            "ENABLE_INTRA_ROUND_MARGIN_CHECKING": True,  # Enable margin checking
            "STOCKS": {
                "STOCK_A": {
                    "INITIAL_PRICE": 50.0,  # Low starting price
                    "FUNDAMENTAL_PRICE": 150.0,  # HIGH fundamental - buyers push hard toward this
                    "REDEMPTION_VALUE": 150.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 10000,  # Plenty of shares for shorting
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 0.0,  # No dividends - clean test
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                },
                "STOCK_B": {
                    "INITIAL_PRICE": 50.0,  # Same starting price
                    "FUNDAMENTAL_PRICE": 150.0,  # HIGH fundamental
                    "REDEMPTION_VALUE": 150.0,
                    "TRANSACTION_COST": 0.0,
                    "LENDABLE_SHARES": 10000,
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 0.0,
                        'dividend_frequency': 1,
                        'dividend_growth': 0.0,
                        'dividend_probability': 0.0,
                        'dividend_variation': 0.0,
                        'destination': 'dividend'
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': True,  # ENABLE SHORT SELLING
                'position_limit': BASE_POSITION_LIMIT * 2,
                'initial_cash': 100000.0,  # MORE CASH to build larger short positions
                'initial_positions': {
                    "STOCK_A": 100,  # Small initial - forces borrowing
                    "STOCK_B": 100
                },
                'max_order_size': 1000,  # Larger orders allowed
                'agent_composition': {
                    'multi_stock_short_seller': 1,  # Builds large short position
                    'multi_stock_squeeze_buyer': 1,  # ACTIVATES round 3: massive squeeze!
                    'multi_stock_market_maker': 2,  # Provides liquidity
                },
                'margin_requirement': 0.5,  # LOW margin (50%) - easy to violate when price spikes!
                'borrow_model': {
                    'rate': 0.0,  # No borrow fees
                    'payment_frequency': 1,
                    'allow_partial_borrows': True
                },
                'interest_model': {
                    'rate': 0.0,  # No interest to keep cash predictable
                    'compound_frequency': 1
                },
                'leverage_params': {
                    'enabled': True,
                    'max_leverage_ratio': 4.0,  # Higher leverage allowed
                    'initial_margin': 0.25,
                    'maintenance_margin': 0.1,
                    'interest_rate': 0.0,
                    'cash_lending_pool': float('inf'),
                    'allow_partial_borrows': True,
                }
            }
        }
    ),

    # =========================================================================
    # DIVIDEND SHOCK SCENARIOS (Issue #86)
    # Test systematic vs idiosyncratic dividend shocks
    # =========================================================================

    "systematic_shock_test": SimulationScenario(
        name="systematic_shock_test",
        description="All stocks receive same systematic dividend shock - tests market-wide correlation",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "IS_MULTI_STOCK": True,
            "HIDE_FUNDAMENTAL_PRICE": False,
            "STOCKS": {
                "STOCK_A": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "style": "common",  # All stocks same style
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.5,
                        'destination': 'dividend',
                        'systematic_beta': 1.0,  # Full exposure to systematic shock
                        'style_gamma': 0.0,  # No style exposure
                        'style': 'common',
                    }
                },
                "STOCK_B": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "style": "common",
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.5,
                        'destination': 'dividend',
                        'systematic_beta': 1.0,
                        'style_gamma': 0.0,
                        'style': 'common',
                    }
                },
                "STOCK_C": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "style": "common",
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.5,
                        'destination': 'dividend',
                        'systematic_beta': 1.0,
                        'style_gamma': 0.0,
                        'style': 'common',
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH * 3,
                'initial_positions': {
                    "STOCK_A": 3333,
                    "STOCK_B": 3333,
                    "STOCK_C": 3334
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'hold_trader': 5  # Passive agents to observe dividend patterns
                },
                # Shock structure configuration
                'shock_structure': {
                    'enabled': True,
                    'systematic_volatility': 1.0,  # Large systematic shocks
                    'styles': {}  # No style-level shocks
                },
                'interest_model': {
                    'rate': 0.05,
                    'compound_frequency': 1
                }
            }
        }
    ),

    "style_shock_test": SimulationScenario(
        name="style_shock_test",
        description="Stocks grouped by style - within-style correlation, cross-style independence",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "IS_MULTI_STOCK": True,
            "HIDE_FUNDAMENTAL_PRICE": False,
            "STOCKS": {
                "TECH_A": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "style": "tech",
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.2,
                        'destination': 'dividend',
                        'systematic_beta': 0.0,  # No systematic exposure
                        'style_gamma': 1.0,  # Full style exposure
                        'style': 'tech',
                    }
                },
                "TECH_B": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "style": "tech",
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.2,
                        'destination': 'dividend',
                        'systematic_beta': 0.0,
                        'style_gamma': 1.0,
                        'style': 'tech',
                    }
                },
                "PHARMA_A": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "style": "pharma",
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.2,
                        'destination': 'dividend',
                        'systematic_beta': 0.0,
                        'style_gamma': 1.0,
                        'style': 'pharma',
                    }
                },
                "PHARMA_B": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "style": "pharma",
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.2,
                        'destination': 'dividend',
                        'systematic_beta': 0.0,
                        'style_gamma': 1.0,
                        'style': 'pharma',
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH * 4,
                'initial_positions': {
                    "TECH_A": 2500,
                    "TECH_B": 2500,
                    "PHARMA_A": 2500,
                    "PHARMA_B": 2500
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'hold_trader': 5
                },
                'shock_structure': {
                    'enabled': True,
                    'systematic_volatility': 0.0,  # No systematic shocks
                    'styles': {
                        'tech': {'volatility': 0.8},
                        'pharma': {'volatility': 0.6}
                    }
                },
                'interest_model': {
                    'rate': 0.05,
                    'compound_frequency': 1
                }
            }
        }
    ),

    "mixed_shock_test": SimulationScenario(
        name="mixed_shock_test",
        description="Full factor structure: systematic + style + idiosyncratic shocks",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 15,
            "IS_MULTI_STOCK": True,
            "HIDE_FUNDAMENTAL_PRICE": False,
            "STOCKS": {
                "TECH_HIGH_BETA": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "style": "tech",
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.3,  # Idiosyncratic
                        'destination': 'dividend',
                        'systematic_beta': 1.5,  # High systematic exposure
                        'style_gamma': 1.0,
                        'style': 'tech',
                    }
                },
                "TECH_LOW_BETA": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "style": "tech",
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.3,
                        'destination': 'dividend',
                        'systematic_beta': 0.5,  # Low systematic exposure
                        'style_gamma': 1.0,
                        'style': 'tech',
                    }
                },
                "PHARMA_HIGH_BETA": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "style": "pharma",
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.3,
                        'destination': 'dividend',
                        'systematic_beta': 1.2,
                        'style_gamma': 1.0,
                        'style': 'pharma',
                    }
                },
                "PHARMA_LOW_BETA": {
                    "INITIAL_PRICE": 100.0,
                    "FUNDAMENTAL_PRICE": 100.0,
                    "REDEMPTION_VALUE": 100.0,
                    "TRANSACTION_COST": 0.0,
                    "style": "pharma",
                    "DIVIDEND_PARAMS": {
                        'type': 'stochastic',
                        'base_dividend': 2.0,
                        'dividend_frequency': 1,
                        'dividend_probability': 0.5,
                        'dividend_variation': 0.3,
                        'destination': 'dividend',
                        'systematic_beta': 0.3,
                        'style_gamma': 1.0,
                        'style': 'pharma',
                    }
                }
            },
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH * 4,
                'initial_positions': {
                    "TECH_HIGH_BETA": 2500,
                    "TECH_LOW_BETA": 2500,
                    "PHARMA_HIGH_BETA": 2500,
                    "PHARMA_LOW_BETA": 2500
                },
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'hold_trader': 5
                },
                'shock_structure': {
                    'enabled': True,
                    'systematic_volatility': 0.5,
                    'styles': {
                        'tech': {'volatility': 0.4},
                        'pharma': {'volatility': 0.3}
                    }
                },
                'interest_model': {
                    'rate': 0.05,
                    'compound_frequency': 1
                }
            }
        }
    ),
}
