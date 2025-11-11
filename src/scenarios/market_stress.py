"""
Market Stress Scenarios

Testing market behavior under stressed conditions including:
- Limited liquidity
- Divergent beliefs about fundamental values
- Opposing optimistic and pessimistic views
"""

from .base import (
    SimulationScenario, DEFAULT_PARAMS,
    FUNDAMENTAL_WITH_DEFAULT_PARAMS, BASE_NUM_ROUNDS, BASE_INITIAL_CASH,
    BASE_INITIAL_SHARES, BASE_MAX_ORDER_SIZE, BASE_POSITION_LIMIT
)

SCENARIOS = {
    "liquidity_crisis": SimulationScenario(
        name="liquidity_crisis",
        description="Testing market resilience under limited liquidity",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 75,
            "INITIAL_PRICE": 28.0,
            # Fundamental calculated as E(d)/r
            "INITIAL_CASH": 0.5 * BASE_INITIAL_CASH,  # Lower initial cash
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': 50000000,  # Lower position limit
                'initial_cash': 0.5 * BASE_INITIAL_CASH,    # Lower initial cash
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'momentum': 3,           # Trend followers
                    'market_maker_sell': 1,  # Liquidity provider with sell bias
                    'contrarian': 2          # Potential stabilizing force
                },
                'deterministic_params': {
                    'gap_trader': {
                        'threshold': 0.05,
                        'max_proportion': 0.5,
                        'scaling_factor': 2.0
                    }
                },
                'type_specific_params': {
                    'market_maker_sell': {
                        'initial_cash': 20.0 * BASE_INITIAL_CASH,  # 2x scenario default
                        'initial_shares': int(20.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'momentum': {
                        'initial_cash': 0.3 * BASE_INITIAL_CASH,   # Lower than default
                        'initial_shares': int(0.8 * BASE_INITIAL_SHARES)      # Lower than default
                    },
                    'contrarian': {
                        'initial_cash': 0.7 * BASE_INITIAL_CASH,   # Higher than momentum
                        'initial_shares': int(1.5 * BASE_INITIAL_SHARES)     # Higher than momentum
                    }
                }
            }
        }
    ),

    "market_stress": SimulationScenario(
        name="market_stress",
        description="Testing market behavior with opposing views",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 100,
            "INITIAL_PRICE": 28.0,
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'position_limit': 100000000,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'optimistic': 2,    # Believes price should be higher
                    'pessimistic': 2,   # Believes price should be lower
                    'market_maker': 2,  # Liquidity providers
                    'value': 2          # Rational anchor
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 2.0 * BASE_INITIAL_CASH,  # 2x default
                        'initial_shares': int(2.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'optimistic': {
                        'initial_cash': 1.5 * BASE_INITIAL_CASH,  # Higher resources to push prices up
                        'initial_shares': int(0.5 * BASE_INITIAL_SHARES)      # Fewer shares (wants to buy more)
                    },
                    'pessimistic': {
                        'initial_cash': 0.5 * BASE_INITIAL_CASH,   # Less cash (wants to sell)
                        'initial_shares': int(1.5 * BASE_INITIAL_SHARES)     # More shares to sell
                    },
                    'value': {
                        'initial_cash': 1.0 * BASE_INITIAL_CASH,  # Balanced
                        'initial_shares': int(1.0 * BASE_INITIAL_SHARES)     # Balanced
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

    "test_hidden_fundamental": SimulationScenario(
        name="test_hidden_fundamental",
        description="Testing agent behavior when fundamental value is hidden",
        parameters={
            **DEFAULT_PARAMS,  # Include defaults
            "NUM_ROUNDS": 3,  # Short test with just 3 rounds
            "HIDE_FUNDAMENTAL_PRICE": True,  # Hide fundamental value
            "INITIAL_PRICE": 28.0,
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': 100000000,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 1,        # Value investor (relies on fundamentals)
                    'momentum': 1,     # Momentum trader (relies on price trends)
                    'market_maker': 1  # Market maker (provides liquidity)
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

    "divergent_beliefs_above_fundamental": SimulationScenario(
        name="divergent_beliefs_above_fundamental",
        description="Testing market with agents having different beliefs about fundamental value",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": BASE_NUM_ROUNDS,
            "HIDE_FUNDAMENTAL_PRICE": True,  # Hide actual fundamental
            "INITIAL_PRICE": 2*FUNDAMENTAL_WITH_DEFAULT_PARAMS,
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': int(0.5 * BASE_INITIAL_SHARES),
                    'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'optimistic': 2,    # Believes fundamental is much higher
                    'pessimistic': 2,   # Believes fundamental is much lower
                    'market_maker': 2,  # Neutral liquidity providers
                    'momentum': 2,      # Doesn't care about fundamentals
                    'default': 2        # Default investors
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 20.0 * BASE_INITIAL_CASH,  # 2x default
                        'initial_shares': int(20.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'speculator': {
                        'initial_cash': 10.0 * BASE_INITIAL_CASH,  # 1.5x default
                        'initial_shares': int(10.0 * BASE_INITIAL_SHARES)      # Half default
                    },
                    'default': {
                        'initial_cash': 1.0 * BASE_INITIAL_CASH,  # Default
                        'initial_shares': int(1.0 * BASE_INITIAL_SHARES)     # Default
                    },
                    'optimistic': {
                        'initial_cash': 1.2 * BASE_INITIAL_CASH,  # 1.2x default
                        'initial_shares': int(1.2 * BASE_INITIAL_SHARES)     # 1.2x default
                    }
                }
            }
        }
    ),
    "divergent_beliefs_below_fundamental": SimulationScenario(
        name="divergent_beliefs_below_fundamental",
        description="Testing market with agents having different beliefs about fundamental value",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": BASE_NUM_ROUNDS,
            "HIDE_FUNDAMENTAL_PRICE": True,  # Hide actual fundamental
            "INITIAL_PRICE": round(FUNDAMENTAL_WITH_DEFAULT_PARAMS/2, 2),
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': True,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': int(0.5 * BASE_INITIAL_SHARES),
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'optimistic': 2,    # Believes fundamental is much higher
                    'pessimistic': 2,   # Believes fundamental is much lower
                    'market_maker': 2,  # Neutral liquidity providers
                    'momentum': 2,      # Doesn't care about fundamentals
                    'default': 2        # Default investors
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 20.0 * BASE_INITIAL_CASH,  # 2x default
                        'initial_shares': int(20.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'optimistic': {
                        'initial_cash': 10.0 * BASE_INITIAL_CASH,  # More cash to buy
                        'initial_shares': int(10.0 * BASE_INITIAL_SHARES)      # Fewer shares
                    },
                    'pessimistic': {
                        'initial_cash': 0.5 * BASE_INITIAL_CASH,   # Less cash
                        'initial_shares': int(10.0 * BASE_INITIAL_SHARES)     # More shares to sell
                    },
                    'value': {
                        'initial_cash': 1.0 * BASE_INITIAL_CASH,  # Balanced
                        'initial_shares': int(1.0 * BASE_INITIAL_SHARES)     # Balanced
                    }
                }
            }
        }
    ),
}
