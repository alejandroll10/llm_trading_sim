"""Regression tests for InformationService signal delay (issues #98, #99).

#98: `_modify_signal` serves a stale VALUE from `signal_history[round - delay]`
     while keeping the current round's structural metadata.
#99: combining a MARKET-category signal (`order_book`, whose value is a dict)
     with `delay > 0` must NOT alias the dict object stored in a past round's
     `signal_history['base']` entry, even when no `depth` limit is configured.
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


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
