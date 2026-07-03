"""Regression tests for InformationService signal delay (issues #98, #99, #100).

#98: `_modify_signal` serves a stale VALUE from `signal_history[round - delay]`
     while keeping the current round's structural metadata.
#99: combining a MARKET-category signal (`order_book`, whose value is a dict)
     with `delay > 0` must NOT alias the dict object stored in a past round's
     `signal_history['base']` entry, even when no `depth` limit is configured.
#100: automated assertions (no more manual log inspection) that, for the delayed
     agent, the served FUNDAMENTAL signal carries the stale round's VALUE but the
     CURRENT round's structural metadata (`current_round`, `is_stale`,
     `original_round`, `periods_remaining`) — covering both the single-stock and
     the multi-stock per-stock lookup path in `_get_delayed_base_signal`.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.market.information.base_information_services import InformationService
from src.market.information.information_types import (
    InfoCapability,
    InformationSignal,
    InformationType,
)


class _StubAgentRepository:
    """Minimal stand-in; `_modify_signal` never touches the repository."""

    def get_all_agent_ids(self):
        return []

    def get_agent(self, agent_id):
        return None


def _make_service():
    # Passing no market_state_managers keeps the service in single-stock mode,
    # matching the `history['base'][info_type]` layout the test populates.
    return InformationService(agent_repository=_StubAgentRepository())


def _order_book_signal(value):
    return InformationSignal(
        type=InformationType.ORDER_BOOK,
        value=value,
        reliability=1.0,
    )


def test_delayed_order_book_does_not_alias_history():
    """order_book + delay (no depth) must serve a copy, not the stored dict."""
    svc = _make_service()

    # Round 0 base signal: the dict that lives in signal_history.
    stored_value = {
        'buy_levels': [{'price': 100, 'quantity': 5}],
        'sell_levels': [{'price': 101, 'quantity': 3}],
    }
    svc.signal_history[0] = {
        'base': {InformationType.ORDER_BOOK: _order_book_signal(stored_value)},
        'agent': {},
    }

    # Current (round 2) signal has a different, fresh value.
    current_signal = _order_book_signal({'buy_levels': [], 'sell_levels': []})

    # delay=2 with no depth limit -> serve round-0 value.
    capability = InfoCapability(enabled=True, delay=2, depth=None)
    result = svc._modify_signal(current_signal, capability, round_number=2)

    # Correct content was served (the stale round-0 value)...
    assert result.value == stored_value
    assert result.metadata['is_stale'] is True
    assert result.metadata['original_round'] == 0

    # ...but it must be a distinct object, not an alias into history.
    assert result.value is not stored_value
    # The nested level lists must also be fresh objects (no depth limit set).
    assert result.value['buy_levels'] is not stored_value['buy_levels']
    assert result.value['sell_levels'] is not stored_value['sell_levels']

    # Mutating the served value (both top-level and via list append) must not
    # corrupt the stored history dict.
    result.value['injected'] = True
    result.value['buy_levels'].append({'price': 0, 'quantity': 999})
    assert stored_value['buy_levels'] == [{'price': 100, 'quantity': 5}]
    assert 'injected' not in stored_value


def test_delayed_order_book_with_depth_truncates_copy():
    """depth still truncates, and still on a copy (existing behavior preserved)."""
    svc = _make_service()
    stored_value = {
        'buy_levels': [{'price': 100}, {'price': 99}, {'price': 98}],
        'sell_levels': [{'price': 101}, {'price': 102}, {'price': 103}],
    }
    svc.signal_history[0] = {
        'base': {InformationType.ORDER_BOOK: _order_book_signal(stored_value)},
        'agent': {},
    }
    current_signal = _order_book_signal({'buy_levels': [], 'sell_levels': []})

    capability = InfoCapability(enabled=True, delay=2, depth=1)
    result = svc._modify_signal(current_signal, capability, round_number=2)

    assert result.value['buy_levels'] == [{'price': 100}]
    assert result.value['sell_levels'] == [{'price': 101}]
    # Original history untouched by truncation.
    assert len(stored_value['buy_levels']) == 3


# ---------------------------------------------------------------------------
# Issue #100: automated FUNDAMENTAL delay-metadata regression.
#
# These tests replace the previous "run the scenario and read the logs by hand"
# verification for `asymmetric_delayed_signal_test`. They drive the real
# distribution path (`distribute_information` -> `_generate_agent_signals` ->
# `_modify_signal` -> `_get_delayed_base_signal`) across several rounds and
# assert the delay invariants directly, for both the single-stock and the
# multi-stock per-stock history layouts.
# ---------------------------------------------------------------------------

TOTAL_ROUNDS = 6


def _fundamental_value(round_number, offset):
    """A per-round, per-stock FUNDAMENTAL value that is unique to its round.

    Distinct values make staleness unambiguous: if the delayed agent is served
    round ``R``'s value instead of ``R - 1``'s, the assertion catches it.
    """
    return offset * 1000 + round_number


def _fundamental_signal(round_number, offset):
    """Mimic FundamentalProvider: value tracks the round, metadata is structural.

    ``periods_remaining`` is computed from the CURRENT round, so the tests can
    verify that delay staling the value never staled this structural field.
    """
    return InformationSignal(
        type=InformationType.FUNDAMENTAL,
        value=_fundamental_value(round_number, offset),
        reliability=1.0,
        metadata={
            'round': round_number,
            'periods_remaining': TOTAL_ROUNDS - round_number,
            'redemption_value': 14.0,
        },
    )


class _FundamentalProvider:
    """Single-stock stand-in for FundamentalProvider (offset 0)."""

    def generate_signal(self, round_number):
        return _fundamental_signal(round_number, offset=0)


class _MultiStockManager:
    """Carries a per-stock value offset so each stock has its own value stream."""

    def __init__(self, offset):
        self.offset = offset


class _MultiStockFundamentalProvider:
    """Multi-stock stand-in: value stream depends on the stock's manager."""

    def generate_signal_for_manager(self, manager, round_number):
        return _fundamental_signal(round_number, offset=manager.offset)


class _StubAgent:
    def __init__(self, capability):
        self.info_capabilities = {InformationType.FUNDAMENTAL: capability}

    def get_info_capability(self, info_type):
        return self.info_capabilities.get(info_type)


class _AgentRepository:
    def __init__(self, agents):
        self._agents = agents

    def get_all_agent_ids(self):
        return list(self._agents.keys())

    def get_agent(self, agent_id):
        return self._agents[agent_id]

    def distribute_information(self, current_signals):
        # Distribution sink is irrelevant here; the tests read `signal_history`.
        pass


def _delayed_and_control_agents(delay=1):
    return _AgentRepository({
        'delayed': _StubAgent(InfoCapability(enabled=True, delay=delay)),
        'control': _StubAgent(InfoCapability(enabled=True, delay=0)),
    })


def test_single_stock_delay_metadata_across_rounds():
    """Drive the single-stock path over several rounds; assert delay invariants.

    For the delayed agent (delay=1) at each round R >= 1:
      value == round (R-1)'s value      (stale value served)
      metadata['original_round'] == R-1
      metadata['current_round'] == R
      metadata['is_stale'] is True
      metadata['periods_remaining'] == TOTAL_ROUNDS - R  (CURRENT round, not stale)
    Round 0 falls back fresh (no history yet): is_stale is False.
    The control agent (delay=0) always sees the current round's value.
    """
    repo = _delayed_and_control_agents(delay=1)
    svc = InformationService(agent_repository=repo)
    svc.register_provider(InformationType.FUNDAMENTAL, _FundamentalProvider())

    for R in range(TOTAL_ROUNDS):
        svc.distribute_information(R)

        delayed = svc.get_signal_history(R, 'delayed')[InformationType.FUNDAMENTAL]
        control = svc.get_signal_history(R, 'control')[InformationType.FUNDAMENTAL]

        # Control always sees the fresh, current value.
        assert control.value == _fundamental_value(R, offset=0)

        # Structural metadata always tracks the CURRENT round for both agents,
        # even though the delayed agent's VALUE may be stale.
        assert delayed.metadata['periods_remaining'] == TOTAL_ROUNDS - R
        assert delayed.metadata['redemption_value'] == 14.0

        if R == 0:
            # No round -1 history: fresh fallback, flagged non-stale.
            assert delayed.value == _fundamental_value(0, offset=0)
            assert delayed.metadata['is_stale'] is False
            assert delayed.metadata['original_round'] == 0
            assert delayed.metadata['current_round'] == 0
        else:
            # Stale VALUE from round R-1, current structural metadata.
            assert delayed.value == _fundamental_value(R - 1, offset=0)
            assert delayed.metadata['is_stale'] is True
            assert delayed.metadata['original_round'] == R - 1
            assert delayed.metadata['current_round'] == R
            # The stale value must NOT drag its own periods_remaining along.
            assert delayed.metadata['periods_remaining'] != TOTAL_ROUNDS - (R - 1)


def test_multi_stock_delay_uses_per_stock_history():
    """Multi-stock: each stock's delayed value comes from ITS OWN history.

    Two stocks with disjoint value streams (offsets 0 and 5). A cross-stock
    lookup bug in `_get_delayed_base_signal` would serve stock A's value for
    stock B (or vice versa); distinct offsets make that detectable.
    """
    repo = _delayed_and_control_agents(delay=1)
    managers = {'A': _MultiStockManager(offset=0), 'B': _MultiStockManager(offset=5)}
    svc = InformationService(agent_repository=repo, market_state_managers=managers)
    svc.register_provider(InformationType.FUNDAMENTAL, _MultiStockFundamentalProvider())
    assert svc.is_multi_stock is True

    for R in range(TOTAL_ROUNDS):
        svc.distribute_information(R)

        agent_signals = svc.get_signal_history(R, 'delayed')
        per_stock = agent_signals['multi_stock_signals']

        for stock_id, offset in (('A', 0), ('B', 5)):
            sig = per_stock[stock_id][InformationType.FUNDAMENTAL]
            assert sig.metadata['periods_remaining'] == TOTAL_ROUNDS - R
            if R == 0:
                assert sig.value == _fundamental_value(0, offset)
                assert sig.metadata['is_stale'] is False
            else:
                # Per-stock stale value: stock A never leaks into stock B.
                assert sig.value == _fundamental_value(R - 1, offset)
                assert sig.metadata['is_stale'] is True
                assert sig.metadata['original_round'] == R - 1
                assert sig.metadata['current_round'] == R


@pytest.mark.parametrize('is_multi_stock', [False, True])
def test_modify_signal_delay_layouts(is_multi_stock):
    """Unit-level `_modify_signal` over both history layouts (issue #100).

    Directly populates round-0 `signal_history` in the single-stock
    (`base[info_type]`) or multi-stock (`base[stock_id][info_type]`) layout,
    then asks `_modify_signal` to serve it at round 1 with delay=1.
    """
    stock_id = 'A' if is_multi_stock else None
    managers = {'A': _MultiStockManager(offset=0)} if is_multi_stock else None
    svc = InformationService(agent_repository=_StubAgentRepository(),
                             market_state_managers=managers)
    assert svc.is_multi_stock is is_multi_stock

    stale_signal = _fundamental_signal(round_number=0, offset=0)
    base_round0 = {InformationType.FUNDAMENTAL: stale_signal}
    svc.signal_history[0] = {
        'base': {stock_id: base_round0} if is_multi_stock else base_round0,
        'agent': {},
    }

    current_signal = _fundamental_signal(round_number=1, offset=0)
    capability = InfoCapability(enabled=True, delay=1)
    result = svc._modify_signal(current_signal, capability, round_number=1,
                                stock_id=stock_id)

    # Stale VALUE (round 0) with CURRENT (round 1) structural metadata.
    assert result.value == _fundamental_value(0, offset=0)
    assert result.metadata['is_stale'] is True
    assert result.metadata['original_round'] == 0
    assert result.metadata['current_round'] == 1
    assert result.metadata['periods_remaining'] == TOTAL_ROUNDS - 1


def test_modify_signal_delay_first_round_falls_back_fresh():
    """delay > 0 with no prior history serves the fresh value, flagged non-stale."""
    svc = InformationService(agent_repository=_StubAgentRepository())
    current_signal = _fundamental_signal(round_number=0, offset=0)
    capability = InfoCapability(enabled=True, delay=1)

    result = svc._modify_signal(current_signal, capability, round_number=0)

    assert result.value == _fundamental_value(0, offset=0)
    assert result.metadata['is_stale'] is False
    assert result.metadata['original_round'] == 0
    assert result.metadata['current_round'] == 0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
