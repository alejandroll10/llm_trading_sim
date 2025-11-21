"""
Scenarios Package

Organized collection of simulation scenarios for testing different
market conditions and agent behaviors.

This package provides a backwards-compatible API for accessing scenarios:
- get_scenario(name): Get a scenario by name
- list_scenarios(): List all available scenarios

Scenarios are organized by category:
- price_discovery: Testing price discovery mechanisms
- market_stress: Testing stressed market conditions
- short_selling: Testing short selling mechanics
- social_dynamics: Testing social influence and herd behavior
- multi_stock: Testing multi-stock trading scenarios
- multi_model: Testing different LLM models competing against each other
- feature_ab_tests: A/B testing pairs for feature toggle system
- test_scenarios: Simple test scenarios for development
- bubbles_professionals: Replicating "Bubbles and Financial Professionals" (Weitzel et al. 2020)
"""

from typing import Dict
from .base import SimulationScenario, DEFAULT_PARAMS

# Import all scenario modules
from . import price_discovery
from . import market_stress
from . import short_selling
from . import social_dynamics
from . import multi_stock
from . import multi_model
from . import feature_ab_tests
from . import test_scenarios
from . import bubbles_professionals

# Combine all scenarios into a single registry
SCENARIOS = {
    **price_discovery.SCENARIOS,
    **market_stress.SCENARIOS,
    **short_selling.SCENARIOS,
    **social_dynamics.SCENARIOS,
    **multi_stock.SCENARIOS,
    **multi_model.MULTI_MODEL_SCENARIOS,
    **feature_ab_tests.SCENARIOS,
    **test_scenarios.SCENARIOS,
    **bubbles_professionals.SCENARIOS,
}

# Backwards-compatible API
def get_scenario(scenario_name: str) -> SimulationScenario:
    """Get a scenario by name"""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available scenarios: {list(SCENARIOS.keys())}")
    return SCENARIOS[scenario_name]

def list_scenarios() -> Dict[str, str]:
    """List all available scenarios and their descriptions"""
    return {name: scenario.description for name, scenario in SCENARIOS.items()}

# Export public API
__all__ = [
    'SimulationScenario',
    'DEFAULT_PARAMS',
    'SCENARIOS',
    'get_scenario',
    'list_scenarios',
]
