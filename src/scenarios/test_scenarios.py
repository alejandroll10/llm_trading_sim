"""
Test Scenarios

Simple, minimal scenarios for testing specific agent types
or features in isolation.
"""

from .base import (
    SimulationScenario, DEFAULT_PARAMS,
    FUNDAMENTAL_WITH_DEFAULT_PARAMS, BASE_NUM_ROUNDS, BASE_INITIAL_CASH,
    BASE_INITIAL_SHARES, BASE_MAX_ORDER_SIZE, BASE_POSITION_LIMIT
)

SCENARIOS = {
    "test_default_trader": SimulationScenario(
        name="test_default_trader",
        description="Testing behavior of default traders in a minimal setting",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 2,  # Short test with just 2 rounds
            "INITIAL_PRICE": 28.0,
            "HIDE_FUNDAMENTAL_PRICE": True,
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'default': 2  # Only 2 default traders
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),
    "test_imbalanced_agents": SimulationScenario(
        name="test_imbalanced_agents",
        description="Testing market with highly imbalanced agent resources",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 3,  # Short test with just 3 rounds
            "INITIAL_PRICE": 28.0,
            "HIDE_FUNDAMENTAL_PRICE": True,
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': 0.01 * BASE_INITIAL_CASH,    # Default initial cash (low)
                'initial_shares': 0.01 * BASE_INITIAL_SHARES,      # Default initial shares (low)
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 1,             # Just one value investor
                    'optimistic': 1       # Just one optimistic investor
                },
                'type_specific_params': {   # Extreme imbalance
                    'optimistic': {
                        'initial_cash': 1.0 * BASE_INITIAL_CASH,  # 100x more cash
                        'initial_shares': int(1.0 * BASE_INITIAL_SHARES)     # 100x more shares
                    }
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),
    "simple_mixed_traders": SimulationScenario(
        name="simple_mixed_traders",
        description="Simple scenario with one market maker, one optimistic trader, and one pessimistic trader",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 3,  # Short scenario with 10 rounds
            "INITIAL_PRICE": FUNDAMENTAL_WITH_DEFAULT_PARAMS,
            "HIDE_FUNDAMENTAL_PRICE": True,
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'market_maker': 1,   # One market maker for liquidity
                    'optimistic': 1,     # One optimistic trader
                    'pessimistic': 1     # One pessimistic trader
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 5.0 * BASE_INITIAL_CASH,
                        'initial_shares': int(5.0 * BASE_INITIAL_SHARES)
                    },
                    'optimistic': {
                        'initial_cash': 1.5 * BASE_INITIAL_CASH,  # More cash to buy
                        'initial_shares': int(0.8 * BASE_INITIAL_SHARES)  # Fewer shares initially
                    },
                    'pessimistic': {
                        'initial_cash': 0.8 * BASE_INITIAL_CASH,  # Less cash
                        'initial_shares': int(1.5 * BASE_INITIAL_SHARES)  # More shares to sell
                    }
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),
    "deterministic_only": SimulationScenario(
        name="deterministic_only",
        description="Scenario with only deterministic agents to test messaging integration",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'margin_requirement': 0.5,
                'borrow_model': {
                    'rate': 0.01,
                    'payment_frequency': 1
                },
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'gap_trader': 1,
                    'momentum_trader': 1,
                    'hold_trader': 1
                }
            }
        }
    ),
    "test_leverage": SimulationScenario(
        name="test_leverage",
        description="Test scenario for leverage trading (2x leverage with margin calls)",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 20,
            "INITIAL_PRICE": 100.0,
            "FUNDAMENTAL_PRICE": 100.0,
            "AGENT_PARAMS": {
                'allow_short_selling': False,  # Focus on long leverage only
                'position_limit': BASE_POSITION_LIMIT * 2,  # Higher limit for leveraged positions
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'momentum_trader': 2,  # Agents that will use leverage
                    'default': 1,  # Agent without leverage for comparison
                },
                'type_specific_params': {
                    'momentum_trader': {
                        'leverage_ratio': 2.0,  # 2x leverage allowed
                        'initial_margin': 0.5,  # 50% down payment required
                        'maintenance_margin': 0.25,  # 25% minimum before margin call
                    },
                    'default': {
                        'leverage_ratio': 1.0,  # No leverage for comparison
                    }
                },
                # NEW: Leverage configuration
                'leverage_params': {
                    'enabled': True,
                    'max_leverage_ratio': 2.0,  # Default max 2x leverage
                    'initial_margin': 0.5,  # 50% down payment
                    'maintenance_margin': 0.25,  # 25% liquidation threshold
                    'interest_rate': 0.05,  # 5% annual interest on borrowed cash
                    'cash_lending_pool': float('inf'),  # Unlimited lending pool
                    'allow_partial_borrows': True,
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                }
            }
        }
    ),
    "test_leverage_llm": SimulationScenario(
        name="test_leverage_llm",
        description="Test LLM leverage traders with undervalued price to encourage buying",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 20.0,  # Significantly below fundamental ($28) to encourage leveraged buying
            "FUNDAMENTAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT * 3,  # Higher limit for leveraged positions
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': int(BASE_INITIAL_SHARES * 0.3),  # Start with fewer shares
                'max_order_size': BASE_MAX_ORDER_SIZE * 2,
                'agent_composition': {
                    'leverage_trader': 2,  # Aggressive leverage users
                    'value': 1,  # Value investor for comparison
                    'market_maker': 1,  # Provides liquidity
                },
                'type_specific_params': {
                    'leverage_trader': {
                        'initial_cash': 0.3 * BASE_INITIAL_CASH,  # LOW cash to force leverage
                        'initial_shares': int(0.1 * BASE_INITIAL_SHARES),  # Very few shares
                        'leverage_ratio': 3.0,  # 3x leverage allowed
                        'initial_margin': 0.5,
                        'maintenance_margin': 0.25,
                    },
                    'value': {
                        'initial_cash': 1.0 * BASE_INITIAL_CASH,
                        'initial_shares': int(0.5 * BASE_INITIAL_SHARES),
                        'leverage_ratio': 1.0,  # No leverage for comparison
                    },
                    'market_maker': {
                        'initial_cash': 2.0 * BASE_INITIAL_CASH,
                        'initial_shares': int(1.5 * BASE_INITIAL_SHARES),
                        'leverage_ratio': 1.0,
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
                },
            }
        }
    ),
}

