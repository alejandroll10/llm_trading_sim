"""
Experiment A1: Information ladder x price anchor (Paper A, Pillar 1)

Implements the A1 base scenario from ``paper/scenario-designs.md`` and issue #103.
Research question: does the *stated valuation* track fundamentals or anchor on
price, by information regime?

Design principle -- elicit, don't prescribe. The workhorse persona is ``default``
(neutral, no strategy hint), so nothing we measure was placed in the prompt. If any
prompt text leaks valuation guidance, that is a bug for this experiment class.

The 15-cell grid over ``FUNDAMENTAL_INFO_MODE`` x ``INITIAL_PRICE`` is NOT encoded
here: it is expressed as a sweep variant pack (``sweeps/variants/a1_info_ladder.json``)
so the same neutral base runs across every cell with matched seeds (same seed => same
dividend path => within-path mode/anchor contrasts). This module only defines the
neutral base scenario that the sweep re-parameterizes per cell.
"""

from .base import (
    SimulationScenario,
    DEFAULT_PARAMS,
    FundamentalInfoMode,
    FUNDAMENTAL_WITH_DEFAULT_PARAMS,
    BASE_INITIAL_CASH,
    BASE_INITIAL_SHARES,
)

# Fundamental value under INFINITE_ROUNDS: FV = E[d] / r = 1.40 / 0.05 = 28.00
# (E[d] = 0.5*(1.4+1.0) + 0.5*(1.4-1.0) = 1.40 from the default DIVIDEND_PARAMS;
#  r = 0.05 from the default INTEREST_MODEL). The three price anchors in the
# variant pack are 0.5x / 1x / 2x this value = 14 / 28 / 56.
FUNDAMENTAL_VALUE = FUNDAMENTAL_WITH_DEFAULT_PARAMS  # 28.0

a1_information_ladder = SimulationScenario(
    name="a1_information_ladder",
    description=(
        "Experiment A1 base: information ladder x price anchor. Neutral default x 8 "
        "population, infinite horizon, 20 rounds. The FUNDAMENTAL_INFO_MODE x "
        "INITIAL_PRICE grid is applied per cell via sweeps/variants/a1_info_ladder.json."
    ),
    parameters={
        **DEFAULT_PARAMS,
        "RANDOM_SEED": 42,
        "NUM_ROUNDS": 20,
        "INFINITE_ROUNDS": True,
        # Neutral base defaults; overridden per sweep cell. FULL is the arithmetic
        # control and INITIAL_PRICE = 1x FV is the un-anchored center of the grid.
        "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.FULL,
        "INITIAL_PRICE": FUNDAMENTAL_VALUE,
        "PROMPT_FAMILY": "a1_default",
        "AGENT_PARAMS": {
            **DEFAULT_PARAMS["AGENT_PARAMS"],
            "allow_short_selling": False,
            "initial_cash": BASE_INITIAL_CASH,
            "initial_shares": BASE_INITIAL_SHARES,
            # Composition: default x 8. No strategy hints anywhere (elicit-don't-prescribe).
            "agent_composition": {
                "default": 8,
            },
        },
    },
)

SCENARIOS = {
    "a1_information_ladder": a1_information_ladder,
}
