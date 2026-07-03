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
    # Regression: DELAYED fundamental signal (issue #98). Agent 1 receives the
    # FUNDAMENTAL signal with a 1-round delay; agent 0 is the undelayed control.
    #
    # The framework enforces a CONSTANT fundamental value by construction
    # (redemption_value = expected_dividend / interest_rate; see
    # scenarios/base.py::_calculate_fundamental_values), so the delay is verified
    # via the signal metadata rather than the value: for the delayed agent,
    # info_signals.log must show, each round R>=1, `original_round == R - 1`,
    # `current_round == R`, and `is_stale == True`, while round 0 falls back to
    # the fresh value (`is_stale == False`, no history yet). Structural metadata
    # (`periods_remaining`) must still track the CURRENT round R, not the stale
    # round, so time-to-redemption stays correct. Deterministic gap traders keep
    # this offline and assertable.
    # ========================================================================
    "asymmetric_delayed_signal_test": SimulationScenario(
        name="asymmetric_delayed_signal_test",
        description=(
            "2 deterministic gap traders; agent 1 receives the FUNDAMENTAL "
            "signal with a 1-round delay, agent 0 is the undelayed control. "
            "Regression for functional signal delay (issue #98)."
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
                    'gap_trader': 2,
                },
                'info_capabilities': {
                    'by_index': {
                        # Agent 1 sees last round's fundamental; agent 0 is fresh.
                        1: {'fundamental': {'delay': 1}},
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
