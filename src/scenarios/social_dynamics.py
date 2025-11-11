"""
Social Dynamics Scenarios

Testing scenarios focused on social influence, herd behavior,
and information cascades in markets.
"""

from .base import (
    SimulationScenario, DEFAULT_PARAMS,
    FUNDAMENTAL_WITH_DEFAULT_PARAMS, BASE_NUM_ROUNDS, BASE_INITIAL_CASH,
    BASE_INITIAL_SHARES, BASE_MAX_ORDER_SIZE, BASE_POSITION_LIMIT
)

SCENARIOS = {
    "cascade_test": SimulationScenario(
        name="cascade_test",
        description="Subset of optimistic agents broadcast messages to test information cascades",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 20,
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
                    'optimistic': 2,
                    'value': 4,
                    'market_maker': 2
                }
            }
        }
    ),
    "social_manipulation": SimulationScenario(
        name="social_manipulation",
        description="Test market manipulation via social media: influencers vs herd followers",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,  # Shortened for testing
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'influencer': 2,        # 2 influencers trying to manipulate
                    'herd_follower': 4,     # 4 herd followers susceptible to influence
                    'value': 2,             # 2 value investors as control
                    'contrarian': 1,        # 1 contrarian to create conflict
                }
            }
        }
    ),
    "echo_chamber": SimulationScenario(
        name="echo_chamber",
        description="Echo chamber effect: mostly herd followers reinforcing each other",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'influencer': 1,        # 1 influencer
                    'herd_follower': 7,     # 7 herd followers (majority)
                    'value': 1,             # 1 rational agent
                }
            }
        }
    ),
}
