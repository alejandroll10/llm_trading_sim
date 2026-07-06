"""Tests for mid-run dividend regime shifts (issue #96).

Covers:
- Piecewise fundamental path math (backward recursion, boundary consistency)
- Redemption convention: K = terminal-regime E[d] / r
- Scenario construction with a regime_schedule (constant-FV assertion branch)
- DividendService switching regimes silently at the scheduled boundary
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _logging_stub
_logging_stub.install()

import random

import pytest

from calculate_fundamental import (
    build_expected_dividend_path,
    calculate_fundamental_path,
    expected_dividend_from_params,
    resolve_regime_params,
    validate_regime_schedule,
    verify_fundamental_path_consistency,
)
from market.state.services.dividend_service import DividendService
from scenarios.base import SimulationScenario, DEFAULT_PARAMS

RATE = 0.05

BASE_DIVIDEND_PARAMS = {
    'type': 'stochastic',
    'base_dividend': 1.4,
    'dividend_frequency': 1,
    'dividend_growth': 0.0,
    'dividend_probability': 0.5,
    'dividend_variation': 1.0,
    'destination': 'dividend',
}


def shifted_params(shift_round=5, **overrides):
    """Dividend params with one scheduled regime shift."""
    params = dict(BASE_DIVIDEND_PARAMS)
    params['regime_schedule'] = [{'round': shift_round, **(overrides or {'base_dividend': 1.0})}]
    return params


# ---------------------------------------------------------------------------
# Path math
# ---------------------------------------------------------------------------

def test_expected_dividend_formula():
    # E[d] = p*(b+v) + (1-p)*(b-v)
    assert expected_dividend_from_params(BASE_DIVIDEND_PARAMS) == pytest.approx(1.4)
    assert expected_dividend_from_params(
        {**BASE_DIVIDEND_PARAMS, 'dividend_probability': 0.3}
    ) == pytest.approx(0.3 * 2.4 + 0.7 * 0.4)


def test_resolve_regime_params_boundaries():
    params = shifted_params(shift_round=5, base_dividend=1.0)
    assert resolve_regime_params(params, 0)['base_dividend'] == 1.4
    assert resolve_regime_params(params, 4)['base_dividend'] == 1.4
    assert resolve_regime_params(params, 5)['base_dividend'] == 1.0
    assert resolve_regime_params(params, 9)['base_dividend'] == 1.0
    # regime_schedule key itself is stripped from the resolved params
    assert 'regime_schedule' not in resolve_regime_params(params, 0)


def test_regime_entries_do_not_stack():
    # Second entry stands alone relative to base params, not on top of the first
    params = dict(BASE_DIVIDEND_PARAMS)
    params['regime_schedule'] = [
        {'round': 3, 'base_dividend': 2.0},
        {'round': 6, 'dividend_probability': 0.3},
    ]
    at_7 = resolve_regime_params(params, 7)
    assert at_7['dividend_probability'] == 0.3
    assert at_7['base_dividend'] == 1.4  # base value, NOT 2.0 from the round-3 entry


def test_constant_regime_reproduces_constant_fundamental():
    expected = [1.4] * 10
    path = calculate_fundamental_path(expected, RATE, terminal_value=1.4 / RATE)
    for fv in path:
        assert fv == pytest.approx(28.0)


def test_piecewise_path_terminal_segment_constant():
    params = shifted_params(shift_round=5, base_dividend=1.0)
    expected = build_expected_dividend_path(params, 10)
    assert expected[:5] == [pytest.approx(1.4)] * 5
    assert expected[5:] == [pytest.approx(1.0)] * 5

    terminal_value = 1.0 / RATE  # 20.0, the redemption convention
    path = calculate_fundamental_path(expected, RATE, terminal_value)

    # Terminal regime segment: constant at the terminal-regime fundamental
    for fv in path[5:]:
        assert fv == pytest.approx(20.0)
    # Pre-shift segment: strictly above 20 (higher dividends), below 28
    # (constant-1.4 value), decreasing toward the boundary
    for fv in path[:5]:
        assert 20.0 < fv < 28.0
    assert all(path[t] > path[t + 1] for t in range(4))

    # No-arbitrage recursion holds everywhere, including across the boundary
    verify_fundamental_path_consistency(path, expected, RATE, terminal_value)


def test_verify_path_consistency_detects_corruption():
    params = shifted_params()
    expected = build_expected_dividend_path(params, 10)
    path = calculate_fundamental_path(expected, RATE, 20.0)
    path[5] += 0.01
    with pytest.raises(ValueError, match="inconsistent at round"):
        verify_fundamental_path_consistency(path, expected, RATE, 20.0)


def test_validate_regime_schedule():
    validate_regime_schedule(BASE_DIVIDEND_PARAMS)  # no schedule: no-op
    validate_regime_schedule(shifted_params(5), num_rounds=10)
    with pytest.raises(ValueError, match="outside the simulation horizon"):
        validate_regime_schedule(shifted_params(10), num_rounds=10)
    with pytest.raises(ValueError, match="non-negative int"):
        validate_regime_schedule(shifted_params(-1))
    with pytest.raises(ValueError, match="duplicate"):
        validate_regime_schedule({
            **BASE_DIVIDEND_PARAMS,
            'regime_schedule': [{'round': 2, 'base_dividend': 1.0},
                                {'round': 2, 'base_dividend': 2.0}],
        })
    with pytest.raises(ValueError, match="'round' key"):
        validate_regime_schedule({
            **BASE_DIVIDEND_PARAMS,
            'regime_schedule': [{'base_dividend': 1.0}],
        })


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def _make_scenario(dividend_params, num_rounds=10, **extra):
    return SimulationScenario(
        name="regime_test",
        description="test",
        parameters={
            **DEFAULT_PARAMS,
            "NUM_ROUNDS": num_rounds,
            "DIVIDEND_PARAMS": dividend_params,
            **extra,
        },
    )


def test_scenario_without_schedule_unchanged():
    scenario = _make_scenario(dict(BASE_DIVIDEND_PARAMS))
    assert scenario.parameters["FUNDAMENTAL_PRICE"] == pytest.approx(28.0)
    assert scenario.parameters["REDEMPTION_VALUE"] == pytest.approx(28.0)
    assert "FUNDAMENTAL_PATH" not in scenario.parameters


def test_scenario_with_regime_schedule():
    scenario = _make_scenario(shifted_params(shift_round=5, base_dividend=1.0))
    params = scenario.parameters

    # Redemption convention: K = terminal-regime E[d] / r = 1.0 / 0.05
    assert params["REDEMPTION_VALUE"] == pytest.approx(20.0)

    path = params["FUNDAMENTAL_PATH"]
    assert len(path) == 10
    assert params["FUNDAMENTAL_PRICE"] == pytest.approx(path[0])
    for fv in path[5:]:
        assert fv == pytest.approx(20.0)
    assert path[0] > path[4] > 20.0


def test_scenario_with_infinite_horizon_regime_schedule():
    scenario = _make_scenario(
        shifted_params(shift_round=5, base_dividend=1.0),
        INFINITE_ROUNDS=True,
    )
    params = scenario.parameters
    assert "REDEMPTION_VALUE" not in params
    # Terminal anchor is the terminal-regime continuation value
    assert params["FUNDAMENTAL_PATH"][-1] == pytest.approx(20.0)


def test_scenario_rejects_shift_outside_horizon():
    with pytest.raises(ValueError, match="outside the simulation horizon"):
        _make_scenario(shifted_params(shift_round=12), num_rounds=10)


def test_registered_regression_scenario():
    from scenarios import get_scenario
    scenario = get_scenario("test_regime_shift")
    params = scenario.parameters
    assert params["REDEMPTION_VALUE"] == pytest.approx(20.0)
    assert len(params["FUNDAMENTAL_PATH"]) == params["NUM_ROUNDS"]
    # Deterministic agents only: the regression run must not require LLM calls
    from agents.deterministic.deterministic_registry import DETERMINISTIC_AGENTS
    for agent_type in params["AGENT_PARAMS"]["agent_composition"]:
        assert agent_type in DETERMINISTIC_AGENTS


# ---------------------------------------------------------------------------
# DividendService regime switching
# ---------------------------------------------------------------------------

def _make_service(dividend_params):
    # agent_repository/logger are only touched when payments are processed,
    # which these tests never do
    return DividendService(
        agent_repository=None,
        logger=None,
        dividend_params=dividend_params,
        redemption_value=20.0,
    )


def test_dividend_service_switches_at_boundary():
    service = _make_service(shifted_params(shift_round=5, base_dividend=1.0,
                                           dividend_probability=0.3))
    for round_number in range(5):
        service.update(round_number)
        assert service.calculator.model['base_dividend'] == 1.4
        assert service.calculator.model['dividend_probability'] == 0.5

    for round_number in range(5, 10):
        service.update(round_number)
        assert service.calculator.model['base_dividend'] == 1.0
        assert service.calculator.model['dividend_probability'] == 0.3

    # get_state reflects the active regime (info modes that reveal the model
    # report it truthfully; REALIZATIONS_ONLY never surfaces it to agents)
    model = service.get_state()['model']
    assert model.base_dividend == 1.0
    assert model.expected_dividend == pytest.approx(0.3 * 2.0 + 0.7 * 0.0)


def test_dividend_service_without_schedule_never_rebuilds():
    service = _make_service(dict(BASE_DIVIDEND_PARAMS))
    calculator = service.calculator
    for round_number in range(10):
        service.update(round_number)
    assert service.calculator is calculator


def test_process_round_end_also_syncs_regime():
    # A shift boundary must apply even if update() were skipped for the round.
    # No redemption value + final round => no payment is processed.
    service = DividendService(
        agent_repository=None, logger=None,
        dividend_params=shifted_params(shift_round=3, base_dividend=2.0),
        redemption_value=None,
    )
    service.process_round_end(round_number=3, is_final_round=True)
    assert service.calculator.model['base_dividend'] == 2.0


def test_regime_shift_silent_by_default():
    service = _make_service(shifted_params(shift_round=5, base_dividend=1.0))
    for round_number in range(10):
        service.update(round_number)
    assert service.get_state()['regime_announcement'] is None


def test_regime_shift_announcement_when_toggled():
    params = shifted_params(shift_round=5, base_dividend=1.0)
    params['announce_regime_shifts'] = True
    service = _make_service(params)

    # Before the shift: no announcement
    for round_number in range(5):
        service.update(round_number)
        assert service.get_state()['regime_announcement'] is None

    # From the shift onward: persistent announcement, 1-indexed round, and
    # the new parameters are not revealed in the text
    for round_number in range(5, 10):
        service.update(round_number)
        announcement = service.get_state()['regime_announcement']
        assert announcement is not None
        assert "round 6" in announcement
        assert "1.0" not in announcement


def test_announcement_flows_to_formatted_prompt():
    from market.information.information_types import InformationSignal, InformationType
    from agents.LLMs.signal_extraction.signal_extractor import SignalExtractor
    from agents.LLMs.services.formatting_services import MarketStateFormatter
    from scenarios.base import FundamentalInfoMode

    signal = InformationSignal(
        type=InformationType.DIVIDEND,
        value=1.0,
        reliability=1.0,
        metadata={
            'yields': {'expected': 5.0, 'max': 10.0, 'min': 0.0, 'last': None},
            'max_dividend': 2.0, 'min_dividend': 0.0,
            'last_paid_dividend': None,
            'next_payment_round': 1, 'should_pay': True,
            'variation': 1.0, 'probability': 50.0,
            'dividend_history': [1.2, 0.4],
            'regime_announcement': "ANNOUNCEMENT: The dividend process changed in round 6.",
        },
    )
    context = SignalExtractor.extract_dividend_context(
        signal, FundamentalInfoMode.REALIZATIONS_ONLY
    )
    text = MarketStateFormatter._format_dividend_info_by_mode(
        context, FundamentalInfoMode.REALIZATIONS_ONLY
    )
    assert "ANNOUNCEMENT: The dividend process changed in round 6." in text

    # Without the announcement the formatted text is unchanged
    signal.metadata['regime_announcement'] = None
    context = SignalExtractor.extract_dividend_context(
        signal, FundamentalInfoMode.REALIZATIONS_ONLY
    )
    text = MarketStateFormatter._format_dividend_info_by_mode(
        context, FundamentalInfoMode.REALIZATIONS_ONLY
    )
    assert "ANNOUNCEMENT" not in text


def test_realized_dividends_respect_active_regime():
    random.seed(123)
    service = _make_service(shifted_params(shift_round=5, base_dividend=1.0,
                                           dividend_variation=0.5))
    for round_number in range(10):
        service.update(round_number)
        realization = service.calculator.calculate_dividend()
        if round_number < 5:
            assert realization.total_dividend in (pytest.approx(0.4), pytest.approx(2.4))
        else:
            assert realization.total_dividend in (pytest.approx(0.5), pytest.approx(1.5))
