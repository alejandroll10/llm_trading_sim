"""Load scenarios from config files (JSON/YAML) — no Python required.

Drop a .yaml/.yml/.json file into src/scenarios/configs/ and it is picked up
by the scenario registry automatically. Each file defines either a single
scenario:

    name: my_scenario
    description: "What this scenario tests"
    parameters:            # deep-merged over DEFAULT_PARAMS (merge_params)
      NUM_ROUNDS: 20
      AGENT_PARAMS:
        agent_composition:
          value: 2
          momentum: 2

or several under a top-level "scenarios" list:

    scenarios:
      - name: variant_a
        description: "..."
        parameters: {...}
      - name: variant_b
        description: "..."
        parameters: {...}

Parameters behave exactly like Python-defined scenarios built with
merge_params(DEFAULT_PARAMS, ...): nested dicts merge, agent_composition and
STOCKS replace wholesale, FUNDAMENTAL_INFO_MODE may be a plain string, and
fundamental/redemption values are computed and validated by
SimulationScenario at load time.
"""
import json
from pathlib import Path
from typing import Dict

from .base import SimulationScenario, DEFAULT_PARAMS, merge_params

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_SUPPORTED_SUFFIXES = {".json", ".yaml", ".yml"}


def _read_config(path: Path) -> dict:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            f"{path.name} is a YAML scenario config, but PyYAML is not "
            f"installed. Install it (pip install pyyaml) or use JSON.") from e
    return yaml.safe_load(text)


def _build_scenario(spec, source: Path) -> SimulationScenario:
    if not isinstance(spec, dict):
        raise ValueError(
            f"{source}: scenario spec must be a mapping, "
            f"got {type(spec).__name__}")
    missing = [key for key in ("name", "description") if not spec.get(key)]
    if missing:
        raise ValueError(f"{source}: scenario spec missing required key(s) {missing}")
    unknown = set(spec) - {"name", "description", "parameters"}
    if unknown:
        raise ValueError(
            f"{source}: unknown top-level key(s) {sorted(unknown)}; "
            f"expected: name, description, parameters")
    return SimulationScenario(
        name=spec["name"],
        description=spec["description"],
        parameters=merge_params(DEFAULT_PARAMS, spec.get("parameters") or {}),
    )


def load_scenario_file(path) -> Dict[str, SimulationScenario]:
    """Load one config file; returns {name: SimulationScenario}."""
    path = Path(path)
    data = _read_config(path)
    if isinstance(data, dict) and "scenarios" in data:
        specs = data["scenarios"]
        if not isinstance(specs, list):
            raise ValueError(f"{path}: 'scenarios' must be a list of scenario specs")
    else:
        specs = [data]

    scenarios: Dict[str, SimulationScenario] = {}
    for spec in specs:
        scenario = _build_scenario(spec, path)
        if scenario.name in scenarios:
            raise ValueError(f"{path}: duplicate scenario name '{scenario.name}'")
        scenarios[scenario.name] = scenario
    return scenarios


def load_config_scenarios(config_dir=CONFIG_DIR) -> Dict[str, SimulationScenario]:
    """Load every config file in `config_dir`; returns {name: SimulationScenario}."""
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        return {}

    scenarios: Dict[str, SimulationScenario] = {}
    for path in sorted(config_dir.iterdir()):
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES or path.name.startswith("_"):
            continue
        for name, scenario in load_scenario_file(path).items():
            if name in scenarios:
                raise ValueError(
                    f"Duplicate scenario name '{name}' in {path} "
                    f"(already defined by another config file)")
            scenarios[name] = scenario
    return scenarios
