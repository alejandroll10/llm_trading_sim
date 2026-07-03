"""
Asymmetric-information scenarios (per-agent private signals).

These scenarios exercise scenario-level ``AGENT_PARAMS['info_capabilities']``,
which gives individual agents noisy / delayed / disabled private views of the
FUNDAMENTAL signal so we can build genuine asymmetric-information designs
(who-picks-off-whom, No-Trade theorem tests, belief-updating studies).

See ``src/market/information/info_capability_config.py`` for the full config
schema.

Usage:
    python src/run_base_sim.py asymmetric_fundamental_regression   # offline (deterministic agents)
    python src/run_base_sim.py asymmetric_fundamental_llm          # LLM agents w/ disclosure
"""

from .base import (
    SimulationScenario, DEFAULT_PARAMS, FundamentalInfoMode,
    BASE_INITIAL_CASH, BASE_INITIAL_SHARES, BASE_MAX_ORDER_SIZE, BASE_POSITION_LIMIT,
)

# Eight distinct fundamental-signal noise levels (fraction of true value),
# assigned to the eight agents in composition order. Agent 0 is perfectly
# informed; agent 7 is the noisiest.
HETEROGENEOUS_NOISE = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30]

SCENARIOS = {
    # ========================================================================
    # Regression: 8 identical agents, heterogeneous fundamental signals.
    # Uses deterministic gap traders so it runs offline (no LLM calls) and can
    # be asserted on: each agent must receive a DIFFERENT realized fundamental
    # value, and those values must be logged to agent_data.csv.
    # ========================================================================
    "asymmetric_fundamental_regression": SimulationScenario(
        name="asymmetric_fundamental_regression",
        description=(
            "8 identical deterministic agents with heterogeneous private "
            "fundamental signals (noise 0%-30%). Regression for per-agent "
            "info_capabilities wiring and realized-signal logging."
        ),
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 5,
            "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.FULL,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'gap_trader': 8,
                },
                # Heterogeneous private signals spread across the 8 agents in order.
                'info_capabilities': {
                    'by_type': {
                        'gap_trader': {
                            'fundamental': {'noise_level': HETEROGENEOUS_NOISE},
                        },
                    },
                },
            },
        },
    ),

    # ========================================================================
    # Regression: an agent with a fully DISABLED signal must still build a valid
    # prompt (value hidden, not a KeyError). Uses hold_llm agents, which build
    # the full LLM prompt context but skip the API call, so this runs offline
    # yet exercises the prompt formatter end-to-end.
    # ========================================================================
    "asymmetric_disabled_signal_test": SimulationScenario(
        name="asymmetric_disabled_signal_test",
        description=(
            "hold_llm agents where some have FUNDAMENTAL and ORDER_BOOK signals "
            "disabled. Regression that disabled signals degrade to 'Unavailable' "
            "in the prompt instead of crashing the formatter."
        ),
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 3,
            "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.FULL,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'hold_llm': 4,
                },
                'info_capabilities': {
                    'by_index': {
                        # Agent 1: no fundamental view. Agent 2: no order-book view.
                        # Agent 3: no dividend view. Agent 0: fully informed control.
                        1: {'fundamental': {'enabled': False}},
                        2: {'order_book': {'enabled': False}},
                        3: {'dividend': {'enabled': False}},
                    },
                },
            },
        },
    ),

    # ========================================================================
    # Research variant: LLM value traders with heterogeneous private signals
    # AND prompt disclosure enabled (each agent is told its own signal quality
    # and the market-wide distribution). Common-knowledge asymmetric info.
    # ========================================================================
    "asymmetric_fundamental_llm": SimulationScenario(
        name="asymmetric_fundamental_llm",
        description=(
            "8 LLM value traders with heterogeneous private fundamental signals "
            "(noise 0%-30%) and prompt disclosure of own + others' signal quality."
        ),
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": 10,
            "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.FULL,
            "AGENT_PARAMS": {
                **DEFAULT_PARAMS["AGENT_PARAMS"],
                'position_limit': BASE_POSITION_LIMIT,
                'initial_cash': BASE_INITIAL_CASH,
                'initial_shares': BASE_INITIAL_SHARES,
                'max_order_size': BASE_MAX_ORDER_SIZE,
                'agent_composition': {
                    'value': 8,
                },
                'info_capabilities': {
                    'by_type': {
                        'value': {
                            'fundamental': {'noise_level': HETEROGENEOUS_NOISE},
                        },
                    },
                    'disclose_signal_quality': True,
                    'disclose_others_quality': True,
                },
            },
        },
    ),
}
