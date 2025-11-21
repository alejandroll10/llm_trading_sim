"""
Bubbles and Financial Professionals (Weitzel et al. 2020) - MVP Implementation

This module implements the experimental scenarios from "Bubbles and Financial Professionals"
(Weitzel, Huber, Huber, Kirchler, Lindner & Rose, 2020, Review of Financial Studies).

The paper tests how cash-to-asset (CA) ratio and trader experience affect bubble formation.

Key Findings from Paper:
- Professionals show 25% bubble frequency vs 58% for students
- High CA ratio (10.2) causes bubbles; Low CA ratio (1.0) prevents them
- Short selling reduces bubbles in high CA environments
- Heterogeneous beliefs (SD of price predictions) drive overpricing

Experimental Treatments:
1. HIGH: CA ratio = 10.2 (high liquidity, bubble-prone)
2. LOW: CA ratio = 1.0 (baseline, bubble-resistant)
3. SHORT: CA ratio = 10.2 + short selling enabled (bubble mitigation)

MVP Approximations:
- Agent types proxy for professional vs student behavior
- Professional mix: value, market_maker, contrarian (analytical, risk-averse)
- Student mix: momentum, optimistic, pessimistic (trend-following, emotional)
- Missing: Belief elicitation (future work), capital injection (future work)

Cash-to-Asset Ratio Calculation:
    CA = Total Cash / (Shares Outstanding × Fundamental Value)

    Paper parameters:
    - FV = 28 Taler
    - HIGH: 8 agents × 5700 Taler = 45,600 total cash
           8 agents × 20 shares = 160 shares
           CA = 45,600 / (160 × 28) = 10.18 ≈ 10.2
    - LOW: 8 agents × 560 Taler = 4,480 total cash
           CA = 4,480 / (160 × 28) = 1.0
"""

from .base import SimulationScenario, DEFAULT_PARAMS

# =============================================================================
# Core Market Parameters (from paper Table 1, page 1313)
# =============================================================================

# Asset parameters
FUNDAMENTAL_VALUE = 28.0  # Constant fundamental value (E[d]/r = 1.4/0.05)
EXPECTED_DIVIDEND = 1.4   # Expected dividend: 50% chance of 2.4, 50% chance of 0.4
INTEREST_RATE = 0.05      # Risk-free interest rate: 5% per round
NUM_ROUNDS = 20           # 20 trading rounds (paper uses 15, but 20 is also tested)
NUM_AGENTS = 8            # 8 agents per market

# Treatment-specific endowments (from paper Table 1)
# HIGH treatment (CA = 10.2)
HIGH_CASH_PER_AGENT = 5700    # Cash endowment per agent
HIGH_SHARES_PER_AGENT = 20    # Share endowment per agent

# LOW treatment (CA = 1.0)
LOW_CASH_PER_AGENT = 560      # Cash endowment per agent
LOW_SHARES_PER_AGENT = 20     # Share endowment per agent

# Dividend parameters (stochastic with equiprobable outcomes)
DIVIDEND_HIGH = 2.4   # 50% probability
DIVIDEND_LOW = 0.4    # 50% probability
DIVIDEND_VARIATION = (DIVIDEND_HIGH - DIVIDEND_LOW) / 2  # = 1.0

# =============================================================================
# Agent Composition Strategies
# =============================================================================

# Professional agent mix (proxy for financial professionals)
# Characteristics: Analytical, value-focused, risk-aware, patient
PROFESSIONAL_AGENT_MIX = {
    'value': 3,           # Fundamental analysis, mean reversion
    'market_maker': 2,    # Liquidity provision, spread trading
    'contrarian': 2,      # Counter-cyclical, fade extremes
    'speculator': 1,      # Opportunistic value plays
}

# Student agent mix (proxy for inexperienced traders)
# Characteristics: Trend-following, emotional, FOMO-driven
STUDENT_AGENT_MIX = {
    'momentum': 3,        # Chase trends, "buy high sell higher"
    'optimistic': 2,      # Overly bullish beliefs
    'pessimistic': 1,     # Overly bearish (but minority)
    'speculator': 2,      # Gambling mentality
}

# College student mix - minimal prompting to let natural behavior emerge
COLLEGE_STUDENT_MIX = {
    'college_student': 8,  # All college students
}

# Mixed agent composition (for testing interaction effects)
MIXED_AGENT_MIX = {
    'value': 2,
    'momentum': 2,
    'market_maker': 2,
    'contrarian': 1,
    'optimistic': 1,
}

# =============================================================================
# Base Parameters for All Bubbles & Professionals Scenarios
# =============================================================================

BASE_BUBBLES_PARAMS = {
    **DEFAULT_PARAMS,

    # Simulation settings
    "RANDOM_SEED": 42,
    "NUM_ROUNDS": NUM_ROUNDS,
    "INFINITE_ROUNDS": False,
    "HIDE_FUNDAMENTAL_PRICE": True,  # Agents don't know true FV

    # Market settings (single stock, no transaction costs)
    "INITIAL_PRICE": FUNDAMENTAL_VALUE,
    "TRANSACTION_COST": 0.0,
    "LENDABLE_SHARES": 0,  # No short selling in HIGH/LOW treatments

    # Dividend model (stochastic, equiprobable)
    "DIVIDEND_PARAMS": {
        'type': 'stochastic',
        'base_dividend': EXPECTED_DIVIDEND,
        'dividend_frequency': 1,  # Every round
        'dividend_growth': 0.0,
        'dividend_probability': 0.5,  # Equiprobable
        'dividend_variation': DIVIDEND_VARIATION,  # ±1.0 from base
        'destination': 'main'  # Goes to tradeable cash (enables CA ratio growth)
    },

    # Interest model (compounded per round)
    "INTEREST_MODEL": {
        'rate': INTEREST_RATE,
        'compound_frequency': 'per_round',
        'destination': 'main'  # Goes to tradeable cash (enables CA ratio growth)
    },

    # LLM model (can be overridden in individual scenarios)
    "MODEL_OPEN_AI": "llama-3.1-70b-instruct",
}

# =============================================================================
# Treatment 1: HIGH CA Ratio (Bubble-Prone)
# =============================================================================

high_ca_professionals = SimulationScenario(
    name="bubbles_high_ca_professionals",
    description="HIGH CA ratio (10.2) with professional agents - tests bubble formation in high liquidity",
    parameters={
        **BASE_BUBBLES_PARAMS,
        "AGENT_PARAMS": {
            **BASE_BUBBLES_PARAMS["AGENT_PARAMS"],
            'initial_cash': HIGH_CASH_PER_AGENT,
            'initial_shares': HIGH_SHARES_PER_AGENT,
            'agent_composition': PROFESSIONAL_AGENT_MIX,
            'allow_short_selling': False,
            'MEMORY_ENABLED': False,
            'SOCIAL_ENABLED': False,
        }
    }
)

high_ca_students = SimulationScenario(
    name="bubbles_high_ca_students",
    description="HIGH CA ratio (10.2) with student agents - tests bubble susceptibility of inexperienced traders",
    parameters={
        **BASE_BUBBLES_PARAMS,
        "AGENT_PARAMS": {
            **BASE_BUBBLES_PARAMS["AGENT_PARAMS"],
            'initial_cash': HIGH_CASH_PER_AGENT,
            'initial_shares': HIGH_SHARES_PER_AGENT,
            'agent_composition': STUDENT_AGENT_MIX,
            'allow_short_selling': False,
            'MEMORY_ENABLED': False,
            'SOCIAL_ENABLED': False,
        }
    }
)

high_ca_college = SimulationScenario(
    name="bubbles_high_ca_college",
    description="HIGH CA ratio (10.2) with college student agents - minimal prompt, emergent behavior",
    parameters={
        **BASE_BUBBLES_PARAMS,
        "AGENT_PARAMS": {
            **BASE_BUBBLES_PARAMS["AGENT_PARAMS"],
            'initial_cash': HIGH_CASH_PER_AGENT,
            'initial_shares': HIGH_SHARES_PER_AGENT,
            'agent_composition': COLLEGE_STUDENT_MIX,
            'allow_short_selling': False,
            'MEMORY_ENABLED': False,
            'SOCIAL_ENABLED': False,
        }
    }
)

# =============================================================================
# Treatment 2: LOW CA Ratio (Benchmark)
# =============================================================================

low_ca_professionals = SimulationScenario(
    name="bubbles_low_ca_professionals",
    description="LOW CA ratio (1.0) with professional agents - baseline condition, bubble-resistant",
    parameters={
        **BASE_BUBBLES_PARAMS,
        "AGENT_PARAMS": {
            **BASE_BUBBLES_PARAMS["AGENT_PARAMS"],
            'initial_cash': LOW_CASH_PER_AGENT,
            'initial_shares': LOW_SHARES_PER_AGENT,
            'agent_composition': PROFESSIONAL_AGENT_MIX,
            'allow_short_selling': False,
            'MEMORY_ENABLED': False,
            'SOCIAL_ENABLED': False,
        }
    }
)

low_ca_students = SimulationScenario(
    name="bubbles_low_ca_students",
    description="LOW CA ratio (1.0) with student agents - baseline for inexperienced traders",
    parameters={
        **BASE_BUBBLES_PARAMS,
        "AGENT_PARAMS": {
            **BASE_BUBBLES_PARAMS["AGENT_PARAMS"],
            'initial_cash': LOW_CASH_PER_AGENT,
            'initial_shares': LOW_SHARES_PER_AGENT,
            'agent_composition': STUDENT_AGENT_MIX,
            'allow_short_selling': False,
            'MEMORY_ENABLED': False,
            'SOCIAL_ENABLED': False,
        }
    }
)

# =============================================================================
# Treatment 3: SHORT (High CA + Short Selling)
# =============================================================================

# For short selling treatment, we need lendable shares
# Paper allows unlimited short selling, so set lendable_shares = total shares
TOTAL_SHARES = NUM_AGENTS * HIGH_SHARES_PER_AGENT  # 8 * 20 = 160

short_professionals = SimulationScenario(
    name="bubbles_short_professionals",
    description="HIGH CA ratio (10.2) + short selling with professionals - tests bubble mitigation",
    parameters={
        **BASE_BUBBLES_PARAMS,
        "LENDABLE_SHARES": TOTAL_SHARES,  # Enable short selling
        "AGENT_PARAMS": {
            **BASE_BUBBLES_PARAMS["AGENT_PARAMS"],
            'initial_cash': HIGH_CASH_PER_AGENT,
            'initial_shares': HIGH_SHARES_PER_AGENT,
            'agent_composition': PROFESSIONAL_AGENT_MIX,
            'allow_short_selling': True,  # Enable short selling
            'margin_requirement': 0.5,  # 50% margin (paper standard)
            'borrow_model': {
                'rate': 0.01,  # 1% borrowing cost per round
                'payment_frequency': 1,
                'allow_partial_borrows': True
            },
            'MEMORY_ENABLED': False,
            'SOCIAL_ENABLED': False,
        }
    }
)

short_students = SimulationScenario(
    name="bubbles_short_students",
    description="HIGH CA ratio (10.2) + short selling with students - tests if inexperienced traders use shorts effectively",
    parameters={
        **BASE_BUBBLES_PARAMS,
        "LENDABLE_SHARES": TOTAL_SHARES,
        "AGENT_PARAMS": {
            **BASE_BUBBLES_PARAMS["AGENT_PARAMS"],
            'initial_cash': HIGH_CASH_PER_AGENT,
            'initial_shares': HIGH_SHARES_PER_AGENT,
            'agent_composition': STUDENT_AGENT_MIX,
            'allow_short_selling': True,
            'margin_requirement': 0.5,
            'borrow_model': {
                'rate': 0.01,
                'payment_frequency': 1,
                'allow_partial_borrows': True
            },
            'MEMORY_ENABLED': False,
            'SOCIAL_ENABLED': False,
        }
    }
)

# =============================================================================
# Mixed Treatment (For Exploratory Analysis)
# =============================================================================

high_ca_mixed = SimulationScenario(
    name="bubbles_high_ca_mixed",
    description="HIGH CA ratio (10.2) with mixed agent types - tests professional/student interaction",
    parameters={
        **BASE_BUBBLES_PARAMS,
        "AGENT_PARAMS": {
            **BASE_BUBBLES_PARAMS["AGENT_PARAMS"],
            'initial_cash': HIGH_CASH_PER_AGENT,
            'initial_shares': HIGH_SHARES_PER_AGENT,
            'agent_composition': MIXED_AGENT_MIX,
            'allow_short_selling': False,
            'MEMORY_ENABLED': False,
            'SOCIAL_ENABLED': False,
        }
    }
)

# =============================================================================
# Scenario Registry
# =============================================================================

SCENARIOS = {
    # HIGH CA scenarios (bubble-prone)
    'bubbles_high_ca_professionals': high_ca_professionals,
    'bubbles_high_ca_students': high_ca_students,
    'bubbles_high_ca_college': high_ca_college,
    'bubbles_high_ca_mixed': high_ca_mixed,

    # LOW CA scenarios (benchmark)
    'bubbles_low_ca_professionals': low_ca_professionals,
    'bubbles_low_ca_students': low_ca_students,

    # SHORT selling scenarios (bubble mitigation)
    'bubbles_short_professionals': short_professionals,
    'bubbles_short_students': short_students,
}

# =============================================================================
# Usage Notes
# =============================================================================
"""
To run these scenarios:

    python3 src/run_base_sim.py bubbles_high_ca_professionals
    python3 src/run_base_sim.py bubbles_high_ca_students
    python3 src/run_base_sim.py bubbles_low_ca_professionals
    python3 src/run_base_sim.py bubbles_low_ca_students
    python3 src/run_base_sim.py bubbles_short_professionals
    python3 src/run_base_sim.py bubbles_short_students
    python3 src/run_base_sim.py bubbles_high_ca_mixed

Expected Outcomes (based on paper):
- HIGH + professionals: Moderate bubbles (25% frequency)
- HIGH + students: Frequent bubbles (58% frequency)
- LOW + professionals: Rare bubbles (controlled)
- LOW + students: Moderate bubbles (but less than HIGH)
- SHORT + professionals: Reduced bubbles vs HIGH (short selling works)
- SHORT + students: Bubbles may persist (inexperienced shorts)

Verification Steps:
1. Check CA ratio:
   - HIGH: (8 × 5700) / (160 × 28) = 10.18 ≈ 10.2 ✓
   - LOW: (8 × 560) / (160 × 28) = 1.0 ✓

2. Check fundamental value consistency:
   - FV = E[d] / r = 1.4 / 0.05 = 28 ✓
   - Redemption value = FV = 28 ✓

3. Bubble detection (post-processing needed):
   - RDMAX > 0.15 (relative deviation from FV)
   - AMPLITUDE > 0.25 (boom-bust swing)
   - CRASH > 0.15 (price drop from peak)

Missing Features (Future Implementation):
1. Belief elicitation: Agents predict prices at t, t+1, t+2
2. Capital injection: Exogenous cash inflows during simulation
3. CA ratio tracking: Monitor CA ratio evolution over rounds
4. Automated bubble detection: Calculate RDMAX, AMPLITUDE, CRASH metrics
"""
