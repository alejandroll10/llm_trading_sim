"""Tests for scenario/agent registration infrastructure.

Covers merge_params deep-merge semantics, FUNDAMENTAL_INFO_MODE
normalization, scenario auto-discovery (Python modules + config files),
the config-file loader, the unified agent registry, and up-front
agent_composition validation.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _logging_stub
_logging_stub.install()

from scenarios.base import (
    DEFAULT_PARAMS,
    FundamentalInfoMode,
    SimulationScenario,
    merge_params,
    normalize_fundamental_info_mode,
)


# ---------------------------------------------------------------------------
# merge_params
# ---------------------------------------------------------------------------

def test_merge_params_deep_merges_nested_keys():
    merged = merge_params(DEFAULT_PARAMS, {
        "NUM_ROUNDS": 99,
        "AGENT_PARAMS": {"initial_cash": 12345.0},
    })
    assert merged["NUM_ROUNDS"] == 99
    assert merged["AGENT_PARAMS"]["initial_cash"] == 12345.0
    # Sibling defaults survive the nested override
    assert merged["AGENT_PARAMS"]["allow_short_selling"] is False
    assert merged["AGENT_PARAMS"]["leverage_params"]["enabled"] is False
    assert merged["DIVIDEND_PARAMS"]["base_dividend"] == 1.4


def test_merge_params_replaces_agent_composition_wholesale():
    merged = merge_params(DEFAULT_PARAMS, {
        "AGENT_PARAMS": {"agent_composition": {"value": 4}},
    })
    # Default composition types must NOT leak into the override
    assert merged["AGENT_PARAMS"]["agent_composition"] == {"value": 4}


def test_merge_params_does_not_mutate_inputs():
    base = {"A": {"x": 1}, "B": 2}
    overrides = {"A": {"y": 3}}
    merged = merge_params(base, overrides)
    assert base == {"A": {"x": 1}, "B": 2}
    assert overrides == {"A": {"y": 3}}
    merged["A"]["x"] = 999
    assert base["A"]["x"] == 1


def test_merge_params_recurses_multiple_levels():
    merged = merge_params(DEFAULT_PARAMS, {
        "AGENT_PARAMS": {"leverage_params": {"enabled": True}},
    })
    assert merged["AGENT_PARAMS"]["leverage_params"]["enabled"] is True
    # Sibling leverage defaults survive
    assert merged["AGENT_PARAMS"]["leverage_params"]["initial_margin"] == 0.5


# ---------------------------------------------------------------------------
# normalize_fundamental_info_mode
# ---------------------------------------------------------------------------

def test_normalize_coerces_string_to_enum():
    params = {"FUNDAMENTAL_INFO_MODE": "realizations_only"}
    normalize_fundamental_info_mode(params)
    assert params["FUNDAMENTAL_INFO_MODE"] is FundamentalInfoMode.REALIZATIONS_ONLY


def test_normalize_handles_legacy_hide_flag():
    params = {"HIDE_FUNDAMENTAL_PRICE": False}
    normalize_fundamental_info_mode(params)
    assert params["FUNDAMENTAL_INFO_MODE"] is FundamentalInfoMode.FULL
    assert "HIDE_FUNDAMENTAL_PRICE" not in params


def test_normalize_defaults_to_process_only():
    params = {}
    normalize_fundamental_info_mode(params)
    assert params["FUNDAMENTAL_INFO_MODE"] is FundamentalInfoMode.PROCESS_ONLY


# ---------------------------------------------------------------------------
# Scenario auto-discovery
# ---------------------------------------------------------------------------

def test_registry_contains_scenarios_from_all_modules():
    from scenarios import SCENARIOS
    # One representative from several modules, incl. the historically
    # differently-named multi_model export
    for name in ("price_discovery_above_fundamental", "single_basic",
                 "example_yaml_scenario"):
        assert name in SCENARIOS, f"{name} missing from registry"


def test_get_scenario_unknown_name_raises():
    from scenarios import get_scenario
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario("definitely_not_a_scenario")


def test_yaml_scenario_inherits_defaults():
    from scenarios import get_scenario
    scenario = get_scenario("example_yaml_scenario")
    params = scenario.parameters
    assert params["NUM_ROUNDS"] == 10
    assert params["AGENT_PARAMS"]["agent_composition"] == {"value": 2, "momentum": 2}
    # Inherited defaults + computed fundamental
    assert params["INTEREST_MODEL"]["rate"] == 0.05
    assert params["FUNDAMENTAL_PRICE"] == pytest.approx(28.0)
    assert params["REDEMPTION_VALUE"] == pytest.approx(28.0)
    assert params["FUNDAMENTAL_INFO_MODE"] is FundamentalInfoMode.PROCESS_ONLY


# ---------------------------------------------------------------------------
# Config-file loader
# ---------------------------------------------------------------------------

def test_config_loader_single_scenario(tmp_path):
    from scenarios.config_loader import load_config_scenarios
    (tmp_path / "one.json").write_text(
        '{"name": "cfg_one", "description": "d", '
        '"parameters": {"NUM_ROUNDS": 7}}')
    scenarios = load_config_scenarios(tmp_path)
    assert set(scenarios) == {"cfg_one"}
    assert scenarios["cfg_one"].parameters["NUM_ROUNDS"] == 7


def test_config_loader_scenario_list(tmp_path):
    from scenarios.config_loader import load_config_scenarios
    (tmp_path / "many.json").write_text(
        '{"scenarios": ['
        '{"name": "cfg_a", "description": "d", "parameters": {}},'
        '{"name": "cfg_b", "description": "d", "parameters": {}}]}')
    scenarios = load_config_scenarios(tmp_path)
    assert set(scenarios) == {"cfg_a", "cfg_b"}


def test_config_loader_rejects_duplicate_names(tmp_path):
    from scenarios.config_loader import load_config_scenarios
    (tmp_path / "a.json").write_text(
        '{"name": "dup", "description": "d", "parameters": {}}')
    (tmp_path / "b.json").write_text(
        '{"name": "dup", "description": "d", "parameters": {}}')
    with pytest.raises(ValueError, match="Duplicate scenario name"):
        load_config_scenarios(tmp_path)


def test_config_loader_rejects_missing_keys(tmp_path):
    from scenarios.config_loader import load_config_scenarios
    (tmp_path / "bad.json").write_text('{"name": "no_description"}')
    with pytest.raises(ValueError, match="missing required key"):
        load_config_scenarios(tmp_path)


def test_config_loader_rejects_unknown_keys(tmp_path):
    from scenarios.config_loader import load_config_scenarios
    (tmp_path / "bad.json").write_text(
        '{"name": "x", "description": "d", "paramters": {}}')
    with pytest.raises(ValueError, match="unknown top-level key"):
        load_config_scenarios(tmp_path)


# ---------------------------------------------------------------------------
# Unified agent registry
# ---------------------------------------------------------------------------

def test_registry_mirrors_deterministic_types_into_agent_types():
    from agents.registry import known_agent_types, is_deterministic
    from agents.agent_types import AGENT_TYPES
    from agents.deterministic.deterministic_registry import DETERMINISTIC_AGENTS
    for det_type in DETERMINISTIC_AGENTS:
        assert det_type in AGENT_TYPES, (
            f"deterministic type '{det_type}' not mirrored into AGENT_TYPES")
        assert is_deterministic(det_type)
    # After mirroring, every known type (LLM or deterministic) is in AGENT_TYPES
    assert known_agent_types() == set(AGENT_TYPES)
    assert "value" in known_agent_types()
    assert not is_deterministic("value")


def test_resolve_agent_type_exact_match_only():
    from agents.agent_types import resolve_agent_type
    assert resolve_agent_type("value") == "value"
    with pytest.raises(ValueError, match="Unknown agent type"):
        resolve_agent_type("valu")  # prefix of 'value': must NOT silently match


# ---------------------------------------------------------------------------
# agent_composition validation
# ---------------------------------------------------------------------------

def test_validate_agent_composition_rejects_unknown_type():
    from base_sim import validate_agent_composition
    with pytest.raises(ValueError, match="unknown agent type"):
        validate_agent_composition({"value": 2, "optimstic": 1})


def test_validate_agent_composition_rejects_bad_counts():
    from base_sim import validate_agent_composition
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_agent_composition({"value": -1})
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_agent_composition({"value": 2.5})


def test_validate_agent_composition_accepts_valid():
    from base_sim import validate_agent_composition
    validate_agent_composition({"value": 2, "buy_trader": 1, "gap_trader": 0})
