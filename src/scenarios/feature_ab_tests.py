"""
Feature Toggle A/B Testing Scenarios

These scenario pairs allow testing the impact of individual features
(memory, social messaging) on market outcomes by running identical
scenarios with only feature flags changed.

Usage:
    python src/run_base_sim.py social_with_memory
    python src/run_base_sim.py social_without_memory
    # Compare results to measure memory's impact on social dynamics
"""

from .base import (
    SimulationScenario, DEFAULT_PARAMS,
    FUNDAMENTAL_WITH_DEFAULT_PARAMS, BASE_NUM_ROUNDS, BASE_INITIAL_CASH,
    BASE_INITIAL_SHARES, BASE_MAX_ORDER_SIZE, BASE_POSITION_LIMIT
)

SCENARIOS = {
    # ========================================================================
    # A/B Test 1: Memory Impact on Social Manipulation (Llama 3.1 70B)
    # ========================================================================
    "social_with_memory": SimulationScenario(
        name="social_with_memory",
        description="Social manipulation scenario WITH memory - agents learn from experience",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 15,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # BOTH features enabled
                'MEMORY_ENABLED': True,
                'SOCIAL_ENABLED': True,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'influencer': 2,
                    'herd_follower': 4,
                    'value': 2,
                    'contrarian': 1,
                }
            }
        }
    ),

    "social_without_memory": SimulationScenario(
        name="social_without_memory",
        description="Social manipulation scenario WITHOUT memory - agents don't learn",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 15,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # Only SOCIAL enabled, MEMORY disabled
                'MEMORY_ENABLED': False,
                'SOCIAL_ENABLED': True,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'influencer': 2,
                    'herd_follower': 4,
                    'value': 2,
                    'contrarian': 1,
                }
            }
        }
    ),

    # ========================================================================
    # A/B Test 2: Social Messaging Impact on Value Trading
    # ========================================================================
    "value_with_social": SimulationScenario(
        name="value_with_social",
        description="Value trading WITH social messaging - test information cascades",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 20,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # BOTH features enabled
                'MEMORY_ENABLED': True,
                'SOCIAL_ENABLED': True,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 5,
                    'momentum': 3,
                }
            }
        }
    ),

    "value_without_social": SimulationScenario(
        name="value_without_social",
        description="Value trading WITHOUT social messaging - pure price signals",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 20,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # Only MEMORY enabled, SOCIAL disabled
                'MEMORY_ENABLED': True,
                'SOCIAL_ENABLED': False,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 5,
                    'momentum': 3,
                }
            }
        }
    ),

    # ========================================================================
    # A/B Test 3: All Features vs Minimal Features
    # ========================================================================
    "full_features": SimulationScenario(
        name="full_features",
        description="All features enabled - maximum agent capabilities",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 15,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # ALL features enabled
                'MEMORY_ENABLED': True,
                'SOCIAL_ENABLED': True,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 3,
                    'momentum': 2,
                    'market_maker': 2,
                    'contrarian': 1,
                }
            }
        }
    ),

    "minimal_features": SimulationScenario(
        name="minimal_features",
        description="No features enabled - baseline trading only",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 15,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # NO features enabled
                'MEMORY_ENABLED': False,
                'SOCIAL_ENABLED': False,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 3,
                    'momentum': 2,
                    'market_maker': 2,
                    'contrarian': 1,
                }
            }
        }
    ),

    # ========================================================================
    # A/B Test 4: Memory Impact on Momentum Trading
    # ========================================================================
    "momentum_with_memory": SimulationScenario(
        name="momentum_with_memory",
        description="Momentum traders WITH memory - learn patterns",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 25,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # Only MEMORY enabled
                'MEMORY_ENABLED': True,
                'SOCIAL_ENABLED': False,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'momentum': 6,
                    'value': 2,
                }
            }
        }
    ),

    "momentum_without_memory": SimulationScenario(
        name="momentum_without_memory",
        description="Momentum traders WITHOUT memory - memoryless strategies",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 25,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # NO features enabled
                'MEMORY_ENABLED': False,
                'SOCIAL_ENABLED': False,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'momentum': 6,
                    'value': 2,
                }
            }
        }
    ),

    # ========================================================================
    # A/B Test 5: Memory Impact with GPT-OSS 120B (Reasoning Model)
    # ========================================================================
    "gptoss_social_with_memory": SimulationScenario(
        name="gptoss_social_with_memory",
        description="Social manipulation WITH memory - GPT-OSS 120B reasoning model",
        parameters={
            **DEFAULT_PARAMS,
            "MODEL_OPEN_AI": "gpt-oss-120b",  # Override model
            "NUM_ROUNDS": 15,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # BOTH features enabled
                'MEMORY_ENABLED': True,
                'SOCIAL_ENABLED': True,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'influencer': 2,
                    'herd_follower': 4,
                    'value': 2,
                    'contrarian': 1,
                }
            }
        }
    ),

    "gptoss_social_without_memory": SimulationScenario(
        name="gptoss_social_without_memory",
        description="Social manipulation WITHOUT memory - GPT-OSS 120B reasoning model",
        parameters={
            **DEFAULT_PARAMS,
            "MODEL_OPEN_AI": "gpt-oss-120b",  # Override model
            "NUM_ROUNDS": 15,
            "INITIAL_PRICE": 28.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # Only SOCIAL enabled, MEMORY disabled
                'MEMORY_ENABLED': False,
                'SOCIAL_ENABLED': True,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'influencer': 2,
                    'herd_follower': 4,
                    'value': 2,
                    'contrarian': 1,
                }
            }
        }
    ),

    "gptoss_social_mispriced": SimulationScenario(
        name="gptoss_social_mispriced",
        description="Social manipulation WITH memory and MISPRICING - test memory with actual trades",
        parameters={
            **DEFAULT_PARAMS,
            "MODEL_OPEN_AI": "gpt-oss-120b",  # Override model
            "NUM_ROUNDS": 10,  # Reduced for faster testing
            "INITIAL_PRICE": 35.0,  # 25% above fundamental ($28) to generate trades
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # BOTH features enabled
                'MEMORY_ENABLED': True,
                'SOCIAL_ENABLED': True,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'influencer': 2,
                    'herd_follower': 4,
                    'value': 2,
                    'contrarian': 1,
                }
            }
        }
    ),

    # ========================================================================
    # A/B Test 6: Last Reasoning Impact on Memory Quality
    # Test if showing agents their prior reasoning reduces notes_to_self redundancy
    # ========================================================================
    "memory_with_last_reasoning": SimulationScenario(
        name="memory_with_last_reasoning",
        description="Memory WITH last reasoning shown - agents see their prior decision rationale",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 35.0,  # Above fundamental to generate activity
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # Memory + Last Reasoning enabled
                'MEMORY_ENABLED': True,
                'SOCIAL_ENABLED': False,
                'LAST_REASONING_ENABLED': True,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 3,
                    'momentum': 2,
                    'speculator': 2,
                }
            }
        }
    ),

    "memory_without_last_reasoning": SimulationScenario(
        name="memory_without_last_reasoning",
        description="Memory WITHOUT last reasoning - agents must use notes_to_self for continuity",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "INITIAL_PRICE": 35.0,  # Above fundamental to generate activity
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                # Memory enabled, Last Reasoning disabled
                'MEMORY_ENABLED': True,
                'SOCIAL_ENABLED': False,
                'LAST_REASONING_ENABLED': False,

                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 3,
                    'momentum': 2,
                    'speculator': 2,
                }
            }
        }
    ),

    # Quick 3-round test for faster iteration
    "quick_memory_test": SimulationScenario(
        name="quick_memory_test",
        description="Quick 3-round test for memory/last_reasoning features",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 3,
            "INITIAL_PRICE": 35.0,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'MEMORY_ENABLED': True,
                'SOCIAL_ENABLED': False,
                'LAST_REASONING_ENABLED': True,
                'allow_short_selling': False,
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 2,
                    'momentum': 2,
                }
            }
        }
    ),
}
