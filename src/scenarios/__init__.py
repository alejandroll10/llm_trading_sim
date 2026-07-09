"""
Scenarios Package

Organized collection of simulation scenarios for testing different
market conditions and agent behaviors.

Public API:
- get_scenario(name): Get a scenario by name
- list_scenarios(): List all available scenarios
- SCENARIOS: The full name -> SimulationScenario registry

Adding a scenario (no registration edits needed — both paths are
auto-discovered):

1. Python module: create or extend a module in this package that defines a
   module-level ``SCENARIOS`` dict mapping name -> SimulationScenario. Build
   parameters with ``merge_params(DEFAULT_PARAMS, {...overrides...})`` so
   defaults are inherited instead of restated.

2. Config file: drop a .yaml/.yml/.json file into src/scenarios/configs/
   (see config_loader.py for the format).

Scenario names must be unique across all modules and config files; duplicates
raise at import time instead of silently overwriting each other.
"""

import importlib
import pkgutil
from typing import Dict

from .base import SimulationScenario, DEFAULT_PARAMS, merge_params
from .config_loader import load_config_scenarios

# Package modules that are infrastructure, not scenario collections
_NON_SCENARIO_MODULES = {"base", "config_loader"}

SCENARIOS: Dict[str, SimulationScenario] = {}


def _register(module_scenarios: Dict[str, SimulationScenario], source: str):
    for name, scenario in module_scenarios.items():
        if name in SCENARIOS:
            raise ValueError(
                f"Duplicate scenario name '{name}' (redefined in {source}). "
                f"Scenario names must be unique across all scenario modules "
                f"and config files.")
        SCENARIOS[name] = scenario


# Auto-discover scenario modules: any module in this package exporting a
# module-level SCENARIOS dict is registered. No manual import/spread lists.
for _module_info in pkgutil.iter_modules(__path__):
    _name = _module_info.name
    if _name in _NON_SCENARIO_MODULES or _name.startswith("_"):
        continue
    _module = importlib.import_module(f".{_name}", __name__)
    _module_scenarios = getattr(_module, "SCENARIOS", None)
    if _module_scenarios is None:
        continue
    _register(_module_scenarios, f"scenarios/{_name}.py")

# Config-file scenarios (src/scenarios/configs/*.yaml|yml|json)
_register(load_config_scenarios(), "scenarios/configs/")


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
    'merge_params',
    'SCENARIOS',
    'get_scenario',
    'list_scenarios',
]
