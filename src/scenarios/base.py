from typing import Dict, Any
from enum import Enum
from calculate_fundamental import calculate_fundamental_price, calibrate_redemption_value


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
# LLM Configuration (Non-sensitive settings - safe to commit)
# =============================================================================
# Default LLM provider settings
DEFAULT_LLM_BASE_URL = "https://api.ai.it.ufl.edu/v1"  # UF Hypergator endpoint
DEFAULT_LLM_MODEL = "gpt-oss-120b"                            # UF Hypergator reasoning model (120b more reliable than 20b)

# Alternative configurations (comment/uncomment to switch):
# OpenAI:
# DEFAULT_LLM_BASE_URL = None  # None = use OpenAI default
# DEFAULT_LLM_MODEL = "gpt-4o-2024-11-20"

# Other UF Hypergator models:
# DEFAULT_LLM_MODEL = "gpt-oss-20b"              # Reasoning model
# DEFAULT_LLM_MODEL = "gpt-oss-120b"             # Large reasoning model
# DEFAULT_LLM_MODEL = "llama-3.1-8b-instruct"    # Smaller/faster Llama
# =============================================================================

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
        params = self.parameters

        # If using legacy parameter
        if "HIDE_FUNDAMENTAL_PRICE" in params and "FUNDAMENTAL_INFO_MODE" not in params:
            hide = params["HIDE_FUNDAMENTAL_PRICE"]
            if hide:
                params["FUNDAMENTAL_INFO_MODE"] = FundamentalInfoMode.PROCESS_ONLY
            else:
                params["FUNDAMENTAL_INFO_MODE"] = FundamentalInfoMode.FULL
            # Remove legacy param to avoid confusion
            del params["HIDE_FUNDAMENTAL_PRICE"]

        # Ensure we have a valid mode (use default if missing)
        if "FUNDAMENTAL_INFO_MODE" not in params:
            params["FUNDAMENTAL_INFO_MODE"] = FundamentalInfoMode.PROCESS_ONLY

        # Convert string to enum if needed
        mode = params["FUNDAMENTAL_INFO_MODE"]
        if isinstance(mode, str):
            params["FUNDAMENTAL_INFO_MODE"] = FundamentalInfoMode(mode)

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

            # The difference should be very small (floating point precision)
            difference = abs(test_fundamental - constant_fundamental)
            assert difference < 1e-10, f"Fundamental value not constant: {test_fundamental} != {constant_fundamental}"

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
    "AGENT_PARAMS": {
        # Feature toggles for agent capabilities (opt-in by default)
        'MEMORY_ENABLED': False,  # Enable memory notes system (notes_to_self field)
        'SOCIAL_ENABLED': False,  # Enable social media messaging (post_message field)
        'SELF_MODIFY_ENABLED': False,  # Enable self-modification of system prompts (experimental)

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
