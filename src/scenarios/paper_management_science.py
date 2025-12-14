"""
Paper Scenarios: Management Science Submission

This file contains all scenarios used in the paper:
"Can Large Language Models Trade? Testing Financial Theories with LLM Agents"

Each scenario is frozen for reproducibility. Do not modify without
updating the paper version number.

Paper Version: v1 (December 2024)
"""

from .base import (
    SimulationScenario, DEFAULT_PARAMS, FundamentalInfoMode,
    FUNDAMENTAL_WITH_DEFAULT_PARAMS, BASE_NUM_ROUNDS, BASE_INITIAL_CASH,
    BASE_INITIAL_SHARES, BASE_MAX_ORDER_SIZE, BASE_POSITION_LIMIT
)

# =============================================================================
# PAPER CONSTANTS
# =============================================================================
PAPER_VERSION = "v1"
PAPER_RANDOM_SEED = 42
PAPER_LLM_MODEL = "gpt-oss-120b"  # OpenAI open-weight model (August 2025)

# Standard fundamental value: E[D]/r = 1.40/0.05 = 28.00
FUNDAMENTAL_VALUE = FUNDAMENTAL_WITH_DEFAULT_PARAMS  # 28.0

# =============================================================================
# SECTION 1: PRICE DISCOVERY
# =============================================================================
# Tests convergence to fundamental value from mispricing

price_discovery_above = SimulationScenario(
    name="paper_price_discovery_above",
    description="Price discovery: Initial price 2x above fundamental (infinite horizon)",
    parameters={
        **DEFAULT_PARAMS,
        "RANDOM_SEED": PAPER_RANDOM_SEED,
        "NUM_ROUNDS": 15,
        "INFINITE_ROUNDS": True,
        "INITIAL_PRICE": 2 * FUNDAMENTAL_VALUE,  # $56, 2x overvalued
        "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.PROCESS_ONLY,
        "AGENT_PARAMS": {
            **DEFAULT_PARAMS["AGENT_PARAMS"],
            'allow_short_selling': False,
            'initial_cash': BASE_INITIAL_CASH,
            'initial_shares': BASE_INITIAL_SHARES,
            'agent_composition': {
                'default': 2,
                'optimistic': 2,
                'market_maker': 2,
                'speculator': 2,
            },
            'type_specific_params': {
                'market_maker': {
                    'initial_cash': 20 * BASE_INITIAL_CASH,
                    'initial_shares': 20 * BASE_INITIAL_SHARES,
                }
            }
        }
    }
)

price_discovery_below = SimulationScenario(
    name="paper_price_discovery_below",
    description="Price discovery: Initial price 0.5x below fundamental (infinite horizon)",
    parameters={
        **DEFAULT_PARAMS,
        "RANDOM_SEED": PAPER_RANDOM_SEED,
        "NUM_ROUNDS": 15,
        "INFINITE_ROUNDS": True,
        "INITIAL_PRICE": 0.5 * FUNDAMENTAL_VALUE,  # $14, 0.5x undervalued
        "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.PROCESS_ONLY,
        "AGENT_PARAMS": {
            **DEFAULT_PARAMS["AGENT_PARAMS"],
            'allow_short_selling': False,
            'initial_cash': BASE_INITIAL_CASH,
            'initial_shares': BASE_INITIAL_SHARES,
            'agent_composition': {
                'default': 2,
                'optimistic': 2,
                'market_maker': 2,
                'speculator': 2,
            },
            'type_specific_params': {
                'market_maker': {
                    'initial_cash': 20 * BASE_INITIAL_CASH,
                    'initial_shares': 20 * BASE_INITIAL_SHARES,
                }
            }
        }
    }
)

# =============================================================================
# SECTION 2: SOCIAL DYNAMICS & MANIPULATION
# =============================================================================
# Tests emergent manipulation behavior

social_manipulation = SimulationScenario(
    name="paper_social_manipulation",
    description="Social manipulation: Influencers vs herd followers (tests emergent manipulation)",
    parameters={
        **DEFAULT_PARAMS,
        "RANDOM_SEED": PAPER_RANDOM_SEED,
        "NUM_ROUNDS": 15,
        "INITIAL_PRICE": 35.0,  # Slightly overvalued
        "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.PROCESS_ONLY,
        "LENDABLE_SHARES": 50000,
        "AGENT_PARAMS": {
            **DEFAULT_PARAMS["AGENT_PARAMS"],
            'allow_short_selling': True,
            'initial_cash': BASE_INITIAL_CASH,
            'initial_shares': BASE_INITIAL_SHARES,
            'SOCIAL_ENABLED': True,
            'MEMORY_ENABLED': True,
            'agent_composition': {
                'influencer': 2,
                'herd_follower': 4,
                'value': 2,
                'contrarian': 1,
            },
        },
    }
)

# Infinite horizon version - agents use perpetuity valuation naturally
social_manipulation_infinite = SimulationScenario(
    name="paper_social_manipulation_infinite",
    description="Social manipulation with infinite horizon (natural belief divergence)",
    parameters={
        **DEFAULT_PARAMS,
        "RANDOM_SEED": PAPER_RANDOM_SEED,
        "NUM_ROUNDS": 15,
        "INFINITE_ROUNDS": True,  # Uses infinite horizon prompt
        "INITIAL_PRICE": 35.0,  # Slightly overvalued
        "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.PROCESS_ONLY,
        "LENDABLE_SHARES": 50000,
        "AGENT_PARAMS": {
            **DEFAULT_PARAMS["AGENT_PARAMS"],
            'allow_short_selling': True,
            'initial_cash': BASE_INITIAL_CASH,
            'initial_shares': BASE_INITIAL_SHARES,
            'SOCIAL_ENABLED': True,
            'MEMORY_ENABLED': True,
            'agent_composition': {
                'influencer': 2,
                'herd_follower': 4,
                'value': 2,
                'contrarian': 1,
            },
        },
    }
)

emergent_manipulation = SimulationScenario(
    name="paper_emergent_manipulation",
    description="Emergent manipulation: Value investors with social (NO manipulation prompt)",
    parameters={
        **DEFAULT_PARAMS,
        "RANDOM_SEED": PAPER_RANDOM_SEED,
        "NUM_ROUNDS": 15,
        "INITIAL_PRICE": 35.0,  # Overvalued
        "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.PROCESS_ONLY,
        "LENDABLE_SHARES": 50000,
        "AGENT_PARAMS": {
            **DEFAULT_PARAMS["AGENT_PARAMS"],
            'allow_short_selling': True,
            'initial_cash': BASE_INITIAL_CASH,
            'initial_shares': BASE_INITIAL_SHARES,
            'SOCIAL_ENABLED': True,
            'MEMORY_ENABLED': False,
            'agent_composition': {
                'value': 6,  # Only value investors - will they manipulate?
                'market_maker': 2,
            },
        },
    }
)

# Test with neutral profit maximizers - will they discover manipulation?
neutral_manipulation_test = SimulationScenario(
    name="paper_neutral_manipulation",
    description="Neutral profit maximizers + herd followers: Will manipulation emerge without prompting?",
    parameters={
        **DEFAULT_PARAMS,
        "RANDOM_SEED": PAPER_RANDOM_SEED,
        "NUM_ROUNDS": 15,
        "INITIAL_PRICE": 35.0,  # Overvalued - creates trading opportunity
        "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.PROCESS_ONLY,
        "LENDABLE_SHARES": 50000,
        "AGENT_PARAMS": {
            **DEFAULT_PARAMS["AGENT_PARAMS"],
            'allow_short_selling': True,
            'initial_cash': BASE_INITIAL_CASH,
            'initial_shares': BASE_INITIAL_SHARES,
            'SOCIAL_ENABLED': True,
            'MEMORY_ENABLED': False,
            'agent_composition': {
                'profit_maximizer': 4,  # Neutral - only told to maximize profit
                'herd_follower': 4,     # Will react to messages (creates incentive to manipulate)
            },
        },
    }
)

# =============================================================================
# SECTION 3: CORRELATED BEHAVIOR / SYSTEMIC RISK (also demonstrates No-Trade Theorem)
# =============================================================================
# Tests what happens when all agents use same LLM and agree

correlated_crash = SimulationScenario(
    name="paper_correlated_crash",
    description="Correlated behavior: All value investors, overvalued market (systemic risk)",
    parameters={
        **DEFAULT_PARAMS,
        "RANDOM_SEED": PAPER_RANDOM_SEED,
        "NUM_ROUNDS": 10,
        "INITIAL_PRICE": 2 * FUNDAMENTAL_VALUE,  # 2x overvalued
        "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.PROCESS_ONLY,  # Agents calculate FV themselves
        "AGENT_PARAMS": {
            **DEFAULT_PARAMS["AGENT_PARAMS"],
            'allow_short_selling': False,  # No shorts → all want to sell, no buyers
            'initial_cash': BASE_INITIAL_CASH,
            'initial_shares': BASE_INITIAL_SHARES,
            'agent_composition': {
                'value': 8,  # All same → correlated selling
            },
        }
    }
)

# =============================================================================
# SECTION 4: SHORT SELLING & PRICE CORRECTION
# =============================================================================
# Tests whether short sellers help correct mispricing

bubble_with_shorts = SimulationScenario(
    name="paper_bubble_with_shorts",
    description="Bubble with short selling enabled: Can shorts correct overpricing?",
    parameters={
        **DEFAULT_PARAMS,
        "RANDOM_SEED": PAPER_RANDOM_SEED,
        "NUM_ROUNDS": 20,
        "INITIAL_PRICE": 2 * FUNDAMENTAL_VALUE,
        "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.PROCESS_ONLY,
        "AGENT_PARAMS": {
            **DEFAULT_PARAMS["AGENT_PARAMS"],
            'allow_short_selling': True,
            'margin_requirement': 0.5,
            'initial_cash': BASE_INITIAL_CASH,
            'initial_shares': BASE_INITIAL_SHARES,
            'agent_composition': {
                'value': 2,
                'optimistic': 2,
                'short_seller': 2,
                'market_maker': 2,
            },
        },
        "LENDABLE_SHARES": 100000,
        "BORROW_RATE": 0.02,
    }
)

bubble_without_shorts = SimulationScenario(
    name="paper_bubble_without_shorts",
    description="Bubble without short selling: Control condition",
    parameters={
        **DEFAULT_PARAMS,
        "RANDOM_SEED": PAPER_RANDOM_SEED,
        "NUM_ROUNDS": 20,
        "INITIAL_PRICE": 2 * FUNDAMENTAL_VALUE,
        "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.PROCESS_ONLY,
        "AGENT_PARAMS": {
            **DEFAULT_PARAMS["AGENT_PARAMS"],
            'allow_short_selling': False,
            'initial_cash': BASE_INITIAL_CASH,
            'initial_shares': BASE_INITIAL_SHARES,
            'agent_composition': {
                'value': 2,
                'optimistic': 2,
                'pessimistic': 2,  # Matches short_seller count - bearish but can't short
                'market_maker': 2,
            },
        },
    }
)

# =============================================================================
# EXPORT ALL PAPER SCENARIOS
# =============================================================================
SCENARIOS = {
    # Price discovery
    "paper_price_discovery_above": price_discovery_above,
    "paper_price_discovery_below": price_discovery_below,

    # Social dynamics (infinite horizon for natural belief divergence)
    "paper_social_manipulation_infinite": social_manipulation_infinite,

    # Systemic risk / No-Trade Theorem
    "paper_correlated_crash": correlated_crash,

    # Short selling
    "paper_bubble_with_shorts": bubble_with_shorts,
    "paper_bubble_without_shorts": bubble_without_shorts,
}

# List for reproducibility script
PAPER_SCENARIO_NAMES = list(SCENARIOS.keys())
