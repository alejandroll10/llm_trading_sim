"""Tests for the cross-cell leakage guard in aggregate_sweep (issue #120).

A logger leak makes cell k's structured_decisions.csv also hold the rows of the
cells that ran after it, so the aggregated panel silently attributes one
decision to several cells. The guard catches that at aggregation time; these
tests pin down what does and does not count as leakage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _logging_stub
_logging_stub.install()

import pandas as pd

from aggregate_sweep import report_cross_cell_duplicates

FILENAME = "structured_decisions.csv"


def panel(rows):
    return pd.DataFrame(rows, columns=["cell_id", "timestamp", "agent_id", "round", "decision"])


def test_flags_a_row_claimed_by_two_cells():
    leaky = panel([
        ["c1", "2026-07-25 09:00:00", 1, 0, "Buy"],
        ["c1", "2026-07-25 09:30:00", 1, 0, "Sell"],   # leaked from c2
        ["c2", "2026-07-25 09:30:00", 1, 0, "Sell"],
    ])
    assert report_cross_cell_duplicates(leaky, FILENAME) == 1


def test_repeated_key_within_one_cell_is_not_leakage():
    """An agent submitting two orders in a round writes two rows with one key."""
    clean = panel([
        ["c1", "2026-07-25 09:00:00", 1, 0, "Buy"],
        ["c1", "2026-07-25 09:00:00", 1, 0, "Sell"],
        ["c2", "2026-07-25 09:30:00", 1, 0, "Buy"],
    ])
    assert report_cross_cell_duplicates(clean, FILENAME) == 0


def test_same_key_under_different_agents_or_rounds_is_not_leakage():
    clean = panel([
        ["c1", "2026-07-25 09:00:00", 1, 0, "Buy"],
        ["c2", "2026-07-25 09:00:00", 2, 0, "Buy"],   # different agent
        ["c2", "2026-07-25 09:00:00", 1, 1, "Buy"],   # different round
    ])
    assert report_cross_cell_duplicates(clean, FILENAME) == 0


def test_unchecked_and_empty_panels_are_skipped():
    rows = panel([["c1", "2026-07-25 09:00:00", 1, 0, "Buy"]])
    # No identity key registered for this file -> nothing to check.
    assert report_cross_cell_duplicates(rows, "market_data.csv") == 0
    # Key columns absent (a panel from a different CSV shape).
    assert report_cross_cell_duplicates(rows.drop(columns=["agent_id"]), FILENAME) == 0
    assert report_cross_cell_duplicates(panel([]), FILENAME) == 0
