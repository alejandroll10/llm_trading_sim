"""
Price Discovery Scenarios

Testing how markets discover the fundamental price when starting
from mispriced initial conditions (above or below fundamental).
Includes both finite and infinite horizon variations.
"""

from .base import (
    SimulationScenario, DEFAULT_PARAMS,
    FUNDAMENTAL_WITH_DEFAULT_PARAMS, BASE_NUM_ROUNDS, BASE_INITIAL_CASH,
    BASE_INITIAL_SHARES, BASE_MAX_ORDER_SIZE, BASE_POSITION_LIMIT
)

SCENARIOS = {
    "price_discovery_above_fundamental": SimulationScenario(
        name="price_discovery_above_fundamental",
        description="Testing price discovery with initial mispricing",
        parameters={
            **DEFAULT_PARAMS,  # Include defaults
            "NUM_ROUNDS": 5,  # Reduced for faster testing
            "INITIAL_PRICE": round(FUNDAMENTAL_WITH_DEFAULT_PARAMS*1.25, 2),  # Start above fundamental
            # Fundamental will be calculated as E(d)/r = 1.4*0.5/0.05 = 14
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,  # Default initial cash
                'initial_shares': BASE_INITIAL_SHARES,    # Default initial shares
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'default': 2,        # Default investors
                    'optimistic': 2,      # Optimistic traders
                    'market_maker': 2,   # Market makers for liquidity
                    'speculator': 2       # Speculators
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 20.0 * BASE_INITIAL_CASH,  # 2x default
                        'initial_shares': int(20.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'speculator': {
                        # 'initial_cash': 10.0 * BASE_INITIAL_CASH,  # 1.5x default
                        # 'initial_shares': int(10.0 * BASE_INITIAL_SHARES)      # Half default
                    },
                    'default': {
                        # 'initial_cash': 1.0 * BASE_INITIAL_CASH,  # Default
                        # 'initial_shares': int(1.0 * BASE_INITIAL_SHARES)     # Default
                    },
                    'optimistic': {
                        # 'initial_cash': 1.2 * BASE_INITIAL_CASH,  # 1.2x default
                        # 'initial_shares': int(1.2 * BASE_INITIAL_SHARES)     # 1.2x default
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
    "price_discovery_below_fundamental": SimulationScenario(
        name="price_discovery_below_fundamental",
        description="Testing price discovery with initial mispricing",
        parameters={
            **DEFAULT_PARAMS,  # Include defaults
            "NUM_ROUNDS": 5,  # Reduced for faster testing
            "INITIAL_PRICE": round(FUNDAMENTAL_WITH_DEFAULT_PARAMS*0.75, 2),  # Start below fundamental
            # Fundamental will be calculated as E(d)/r = 1.4*0.5/0.05 = 14
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'default': 2,        # Default investors
                    'optimistic': 2,      # Optimistic traders
                    'market_maker': 2,   # Market makers for liquidity
                    'speculator': 2       # Speculators
                },
                'type_specific_params': {
                    'market_maker': {
                        'initial_cash': 20.0 * BASE_INITIAL_CASH,  # 2x default
                        'initial_shares': int(20.0 * BASE_INITIAL_SHARES)     # 2x default
                    },
                    'speculator': {
                        # 'initial_cash': 10.0 * BASE_INITIAL_CASH,  # 1.5x default
                        # 'initial_shares': int(10.0 * BASE_INITIAL_SHARES)      # Half default
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

    "price_discovery_infinite_above_fundamental": SimulationScenario(
        name="price_discovery_infinite_above_fundamental",
        description="Testing price discovery with initial mispricing and infinite rounds",
        parameters={
            **DEFAULT_PARAMS,  # Include defaults
            "NUM_ROUNDS": BASE_NUM_ROUNDS,
            "INFINITE_ROUNDS": True,
            "INITIAL_PRICE": 2*FUNDAMENTAL_WITH_DEFAULT_PARAMS,  # Start above fundamental
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'default': 2,        # Default investors
                    'optimistic': 2,      # Optimistic traders
                    'market_maker': 2,   # Market makers for liquidity
                    'speculator': 2       # Speculators
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
    "price_discovery_infinite_below_fundamental": SimulationScenario(
        name="price_discovery_infinite_below_fundamental",
        description="Testing price discovery with initial mispricing and infinite rounds",
        parameters={
            **DEFAULT_PARAMS,  # Include defaults
            "NUM_ROUNDS": BASE_NUM_ROUNDS,
            "INFINITE_ROUNDS": True,
            "INITIAL_PRICE": round(FUNDAMENTAL_WITH_DEFAULT_PARAMS/2, 2),  # Start below fundamental
            # Fundamental calculated as E(d)/r
            "AGENT_PARAMS": {
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'default': 2,        # Default investors
                    'optimistic': 2,      # Optimistic traders
                    'market_maker': 2,   # Market makers for liquidity
                    'speculator': 2       # Speculators
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
}
