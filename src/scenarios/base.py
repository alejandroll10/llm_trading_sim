import copy
from typing import Dict, Any
from enum import Enum
from calculate_fundamental import (
    calculate_fundamental_price,
    calibrate_redemption_value,
    build_expected_dividend_path,
    calculate_fundamental_path,
    expected_dividend_from_params,
    resolve_regime_params,
    validate_regime_schedule,
    verify_fundamental_path_consistency,
)


class FundamentalInfoMode(str, Enum):
    """
    Controls what information agents receive about fundamental values.

    The mode determines how much of the dividend model, interest rate,
    and redemption value is revealed to agents. This is crucial for
    experiments testing price discovery under uncertainty.

    Modes:
        FULL: Agents see everything including computed fundamental value.
              Use for rational expectations baselines.

        PROCESS_ONLY: Agents see dividend model parameters but not explicit FV.
                      Redemption value hidden. Agents can still compute FV
                      from E[d]/r but must do so themselves.

        REALIZATIONS_ONLY: Agents only see past dividend payments, not the
                           underlying model. Must learn/estimate distribution.
                           Best for bubbles experiments and learning studies.

        AVERAGE: Agents see running average and std dev of past dividends.
                 Summary statistics only, no model parameters.

        NONE: No dividend or fundamental information shown.
              Agents only see price, volume, and their positions.
    """
    FULL = "full"
    PROCESS_ONLY = "process_only"
    REALIZATIONS_ONLY = "realizations_only"
    AVERAGE = "average"
    NONE = "none"

# =============================================================================
# LLM Configuration
# =============================================================================
# Try to load from local config file (gitignored), fall back to defaults
try:
    from llm_config import LLM_BASE_URL, LLM_MODEL
    DEFAULT_LLM_BASE_URL = LLM_BASE_URL
    DEFAULT_LLM_MODEL = LLM_MODEL
except ImportError:
    # Default: UF Hypergator API (works for most users)
    DEFAULT_LLM_BASE_URL = "https://api.ai.it.ufl.edu/v1"
    DEFAULT_LLM_MODEL = "gpt-oss-120b"


def resolve_llm_api_key():
    """Resolve the API key for the configured LLM backend.

    Call this at client-construction time (after load_dotenv()), not at import
    time, so the .env has already been loaded. DeepInfra authenticates with
    DEEPINFRA_TOKEN; every other backend (UF, OpenAI) uses OPENAI_API_KEY.
    Falls back to OPENAI_API_KEY so an unset token doesn't crash construction.
    """
    import os
    if DEFAULT_LLM_BASE_URL and "deepinfra" in DEFAULT_LLM_BASE_URL:
        token = os.environ.get("DEEPINFRA_TOKEN")
        if not token:
            import logging
            logging.getLogger("llm_timing").warning(
                "DEEPINFRA_TOKEN not set; falling back to OPENAI_API_KEY for the "
                "DeepInfra endpoint. This will likely 401 unless that key is a "
                "valid DeepInfra token."
            )
            token = os.environ.get("OPENAI_API_KEY")
        return token
    return os.environ.get("OPENAI_API_KEY")
# =============================================================================

# Keys whose dict values are complete specifications rather than incremental
# tweaks: an override REPLACES the whole value instead of merging into it.
# (Merging an agent_composition override into the default composition would
# silently keep default agent types the scenario meant to drop.)
REPLACE_WHOLESALE_KEYS = {"agent_composition", "STOCKS"}


def merge_params(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge `overrides` onto `base`, returning a new dict.

    Nested dicts merge recursively, so a scenario or sweep variant can tweak a
    single nested key (e.g. AGENT_PARAMS -> initial_cash) without restating the
    whole block — and without silently dropping sibling defaults, which is what
    happens when a scenario rebuilds AGENT_PARAMS from scratch. Non-dict values
    replace. Values under REPLACE_WHOLESALE_KEYS replace entirely even when
    both sides are dicts. Neither input is mutated.

    Typical scenario usage:

        parameters = merge_params(DEFAULT_PARAMS, {
            "NUM_ROUNDS": 20,
            "AGENT_PARAMS": {
                "allow_short_selling": True,
                "agent_composition": {"value": 2, "momentum": 2},
            },
        })
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if (key not in REPLACE_WHOLESALE_KEYS
                and isinstance(merged.get(key), dict)
                and isinstance(value, dict)):
            merged[key] = merge_params(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def normalize_fundamental_info_mode(params: Dict[str, Any]) -> None:
    """Normalize FUNDAMENTAL_INFO_MODE in `params`, in place.

    Handles the legacy HIDE_FUNDAMENTAL_PRICE flag, fills in the default mode,
    and coerces raw strings (e.g. from JSON sweep-variant packs) to the enum.
    Called by SimulationScenario.__init__ and by the sweep-override path in
    run_base_sim, which bypasses scenario construction.
    """
    if "HIDE_FUNDAMENTAL_PRICE" in params and "FUNDAMENTAL_INFO_MODE" not in params:
        hide = params.pop("HIDE_FUNDAMENTAL_PRICE")
        params["FUNDAMENTAL_INFO_MODE"] = (
            FundamentalInfoMode.PROCESS_ONLY if hide else FundamentalInfoMode.FULL
        )

    # Ensure we have a valid mode (use default if missing)
    if "FUNDAMENTAL_INFO_MODE" not in params:
        params["FUNDAMENTAL_INFO_MODE"] = FundamentalInfoMode.PROCESS_ONLY

    # Convert string to enum if needed
    mode = params["FUNDAMENTAL_INFO_MODE"]
    if isinstance(mode, str):
        params["FUNDAMENTAL_INFO_MODE"] = FundamentalInfoMode(mode)


class SimulationScenario:
    """
    Represents a specific simulation scenario with a defined set of parameters.

    This class encapsulates the configuration for a simulation run, including
    its name, description, and all the necessary parameters to initialize
    and run the simulation. It also automatically calculates and validates
    the fundamental value and redemption value based on the provided
    dividend and interest rate parameters.

    Attributes:
        name (str): The unique name of the scenario.
        description (str): A brief description of what the scenario is testing.
        parameters (Dict[str, Any]): A dictionary of parameters for the simulation.
    """
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

        # Handle legacy HIDE_FUNDAMENTAL_PRICE -> FUNDAMENTAL_INFO_MODE conversion
        self._normalize_fundamental_info_mode()

        # Calculate and validate fundamental prices
        self._calculate_fundamental_values()

    def _normalize_fundamental_info_mode(self):
        """Convert legacy HIDE_FUNDAMENTAL_PRICE to FUNDAMENTAL_INFO_MODE if needed."""
        normalize_fundamental_info_mode(self.parameters)

    def _calculate_fundamental_values(self):
        """Calculate and enforce the constant fundamental value principle where:
        fundamental_value = redemption_value = expected_dividend/interest_rate
        """
        params = self.parameters

        # Extract required parameters
        num_rounds = params["NUM_ROUNDS"]
        is_infinite = params.get("INFINITE_ROUNDS", False)

        # Get dividend parameters
        dividend_params = params.get("DIVIDEND_PARAMS", {})

        # Regime schedules (issue #96) produce a piecewise fundamental path
        # instead of a constant fundamental value
        if dividend_params.get("regime_schedule"):
            self._calculate_regime_fundamental_values()
            return

        base_dividend = dividend_params.get("base_dividend", 1.4)
        dividend_probability = dividend_params.get("dividend_probability", 0.5)
        dividend_variation = dividend_params.get("dividend_variation", 0.0)

        # Calculate expected dividend based on the dividend model - use the same formula as DividendService

        expected_dividend = dividend_probability * (base_dividend + dividend_variation) + \
                            (1 - dividend_probability) * (base_dividend - dividend_variation)

        # Get interest rate
        interest_model = params.get("INTEREST_MODEL", {})
        interest_rate = interest_model.get("rate", 0.05)

        # Calculate the constant fundamental value
        constant_fundamental = expected_dividend / interest_rate

        # Update the parameters
        params["FUNDAMENTAL_PRICE"] = constant_fundamental

        # For infinite horizon, no redemption value is needed
        if is_infinite:
            if "REDEMPTION_VALUE" in params:
                del params["REDEMPTION_VALUE"]
        # For finite horizon, set redemption value equal to fundamental value
        else:
            params["REDEMPTION_VALUE"] = constant_fundamental

        # Verify that with these parameters, the fundamental value is constant
        # across all periods (for debugging purposes)
        if not is_infinite:
            # Get the calculated fundamental with these parameters
            test_fundamental = calculate_fundamental_price(
                num_rounds, expected_dividend, interest_rate, constant_fundamental
            )

            # The difference should be very small (floating point precision).
            # Raise (not assert) so the invariant also holds under `python -O`.
            difference = abs(test_fundamental - constant_fundamental)
            if difference >= 1e-10:
                raise ValueError(
                    f"Scenario '{self.name}': fundamental value not constant: "
                    f"{test_fundamental} != {constant_fundamental}"
                )

    def _calculate_regime_fundamental_values(self):
        """Calculate the piecewise fundamental path for scenarios with a
        dividend regime schedule (issue #96).

        With DIVIDEND_PARAMS['regime_schedule'] = [{'round': r, ...overrides}, ...]
        the fundamental value is no longer constant. Conventions:

        - Redemption (finite horizon): K = terminal-regime E[d] / r, so the
          fundamental is constant and equal to K within the terminal regime
          segment, and follows the no-arbitrage recursion before it.
        - FUNDAMENTAL_PRICE is the round-0 value of the path; the full
          per-round path is stored in FUNDAMENTAL_PATH and applied each round
          by the simulation (recorded in market_data.csv).
        - Shifts are NOT announced to agents: under REALIZATIONS_ONLY agents
          can only infer the change from realized dividends. Info modes that
          reveal model parameters will truthfully show the active regime.
        """
        params = self.parameters
        num_rounds = params["NUM_ROUNDS"]
        is_infinite = params.get("INFINITE_ROUNDS", False)
        dividend_params = params["DIVIDEND_PARAMS"]
        interest_rate = params.get("INTEREST_MODEL", {}).get("rate", 0.05)

        validate_regime_schedule(dividend_params, num_rounds)

        expected_dividends = build_expected_dividend_path(dividend_params, num_rounds)
        terminal_expected = expected_dividend_from_params(
            resolve_regime_params(dividend_params, num_rounds - 1)
        )
        # Terminal anchor: redemption value for finite horizons, terminal-regime
        # continuation value for infinite horizons (same number by convention)
        terminal_value = terminal_expected / interest_rate

        path = calculate_fundamental_path(expected_dividends, interest_rate, terminal_value)
        verify_fundamental_path_consistency(path, expected_dividends, interest_rate, terminal_value)

        params["FUNDAMENTAL_PRICE"] = path[0]
        params["FUNDAMENTAL_PATH"] = path

        if is_infinite:
            params.pop("REDEMPTION_VALUE", None)
        else:
            params["REDEMPTION_VALUE"] = terminal_value

# Base constants
FUNDAMENTAL_WITH_DEFAULT_PARAMS = 28.0
BASE_NUM_ROUNDS = 15
BASE_INITIAL_CASH = 1000000.0
BASE_INITIAL_SHARES = 10000
BASE_INITIAL_PRICE = FUNDAMENTAL_WITH_DEFAULT_PARAMS
BASE_MAX_ORDER_SIZE = 1000
BASE_POSITION_LIMIT = 100000000

# Default parameters that can be overridden by specific scenarios
DEFAULT_PARAMS = {
    # Core simulation parameters
    "RANDOM_SEED": 42,
    "NUM_ROUNDS": BASE_NUM_ROUNDS,
    "INFINITE_ROUNDS": False,
    "FUNDAMENTAL_INFO_MODE": FundamentalInfoMode.PROCESS_ONLY,  # Controls what agents see about fundamentals
    # Legacy support: HIDE_FUNDAMENTAL_PRICE is converted to FUNDAMENTAL_INFO_MODE in SimulationScenario
    "NEWS_ENABLED": False,  # LLM-generated market news (requires extra API calls)

    # Market parameters
    "INITIAL_PRICE": FUNDAMENTAL_WITH_DEFAULT_PARAMS,
    "TRANSACTION_COST": 0.0,
    "LENDABLE_SHARES": 0,

    # Agent parameters
    "MODEL_OPEN_AI": DEFAULT_LLM_MODEL,  # Set at top of this file
    # LLM sampling parameters (defaults preserve prior hard-coded behavior).
    # Per-agent-type overrides can be set via AGENT_PARAMS['type_specific_params'][type]
    # using the 'temperature' and 'seed' keys (parallel to per-type 'model').
    "LLM_TEMPERATURE": 0.0,  # Sampling temperature for LLM trading agents
    "LLM_SEED": 42,          # Deterministic sampling seed for LLM trading agents
    # Concurrent agent decisions (LLM calls) per round. Agents decide
    # simultaneously on the same market state, so this only changes wall-clock
    # time, never results. Set to 1 for endpoints that misbehave under
    # concurrent load (serial mode paces requests 500ms apart). The UF
    # Hypergator fallback endpoint historically rate-limited concurrent
    # gpt-oss calls, so it keeps the old paced-serial behavior by default;
    # any explicitly configured backend (llm_config.py) gets parallel.
    "LLM_MAX_CONCURRENCY": 1 if "ai.it.ufl.edu" in (DEFAULT_LLM_BASE_URL or "") else 8,
    # Prompt-family robustness (issue #102): replacement system prompts keyed by
    # agent type (e.g. {"value": "..."}), plus a label identifying which prompt
    # family the run belongs to (the clustering key for inference across runs
    # sharing a prompt family). Both are typically set per sweep variant; see
    # sweeps/variants/ for the shipped packs.
    "SYSTEM_PROMPT_OVERRIDES": {},
    "PROMPT_FAMILY": "baseline",
    "AGENT_PARAMS": {
        # Feature toggles for agent capabilities (opt-in by default)
        'MEMORY_ENABLED': False,  # Enable memory notes system (notes_to_self field)
        'SOCIAL_ENABLED': False,  # Enable social media messaging (post_message field)
        'SELF_MODIFY_ENABLED': False,  # Enable self-modification of system prompts (experimental)
        'CONFIDENCE_ENABLED': False,  # Elicit confidence (0-1) in valuation and price predictions
        'SECOND_ORDER_ENABLED': False,  # Elicit beliefs about other agents' average stated valuation

        'allow_short_selling': False,
        'margin_requirement': 0.5,
        'margin_base': 'cash',  # "cash" or "wealth" - base for margin calculations
        'borrow_model': {
            'rate': 0.01,
            'payment_frequency': 1,
            'allow_partial_borrows': True  # Allows partial share borrows (more realistic market behavior)
        },
        'leverage_params': {
            'max_leverage_ratio': 1.0,  # 1.0 = no leverage by default
            'initial_margin': 0.5,  # 50% down payment required for leveraged positions
            'maintenance_margin': 0.25,  # 25% minimum margin (liquidation threshold)
            'interest_rate': 0.05,  # 5% per-round interest on borrowed cash
            'cash_lending_pool': float('inf'),  # Unlimited lending pool by default
            'allow_partial_borrows': True,
            'enabled': False  # Leverage disabled by default
        },
        'position_limit': BASE_POSITION_LIMIT,
        'initial_cash': BASE_INITIAL_CASH,
        'initial_shares': BASE_INITIAL_SHARES,
        'max_order_size': BASE_MAX_ORDER_SIZE,
        'agent_composition': {
            'value': 2,
            'momentum': 2,
            'market_maker': 2
        },
        'deterministic_params': {
            'gap_trader': {
                'threshold': 0.05,
                'max_proportion': 0.5,
                'scaling_factor': 2.0
            }
        }
    },

    # Interest model parameters
    "INTEREST_MODEL": {
        'rate': 0.05,
        'compound_frequency': 'per_round',
        'destination': 'dividend'
    },

    # Dividend parameters
    "DIVIDEND_PARAMS": {
        'type': 'stochastic',
        'base_dividend': 1.4,
        'dividend_frequency': 1,
        'dividend_growth': 0.0,
        'dividend_probability': 0.5,
        'dividend_variation': 1.0,
        'destination': 'dividend'
    }
}
