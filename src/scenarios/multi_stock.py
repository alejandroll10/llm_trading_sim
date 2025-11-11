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
}
