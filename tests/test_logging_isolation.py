"""Regression tests for cross-run logger isolation (issue #120).

LoggingService.initialize() runs once per simulation, and run_sweep.py executes
every cell of a sweep in ONE process. Loggers are process-global and keyed by
name, so before the fix the second initialize() left the first run's handlers
attached: cell k's files also collected the rows of cells k+1..N, and
aggregate_sweep attributed them to cell k. Loggers created on demand by
get_logger() were additionally cached across runs, so from cell 2 on their
output landed only in cell 1's directory.

The real service is exercised in a subprocess because tests/conftest.py installs
a no-op stub for services.logging_service process-wide.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

# Two runs in one process, as a two-cell sweep drives it. Each run writes one
# marker through a named logger (simulation), an on-demand logger (market_state,
# created by get_logger's fallback branch) and a CSV logger.
DRIVER = '''
import json
from services.logging_service import LoggingService


class _Entry:
    """Stand-in for DecisionLogEntry; log_structured_decision only calls to_csv."""

    def __init__(self, tag):
        self.tag = tag

    def to_csv(self):
        return "decision-" + self.tag


handler_counts = {}
for tag in ("A", "B"):
    LoggingService.initialize("cells/" + tag)
    LoggingService.log_simulation("sim-" + tag)
    LoggingService.get_logger("market_state").info("adhoc-" + tag)
    LoggingService.log_structured_decision(_Entry(tag))
    handler_counts[tag] = {name: len(logger.handlers)
                           for name, logger in LoggingService._loggers.items()}
print(json.dumps(handler_counts))
'''

# filename -> marker written into it by each run
RUN_FILES = {
    "simulation.log": "sim-",
    "market_state.log": "adhoc-",
    "structured_decisions.csv": "decision-",
}


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    """Run DRIVER in a clean directory; returns (logs_root, handler_counts)."""
    workdir = tmp_path_factory.mktemp("logging_isolation")
    proc = subprocess.run(
        [sys.executable, "-c", DRIVER],
        cwd=workdir,
        env=dict(os.environ, PYTHONPATH=str(SRC)),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"driver failed:\n{proc.stdout}\n{proc.stderr}"
    return workdir / "logs", json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("filename,marker", sorted(RUN_FILES.items()))
def test_run_files_hold_only_their_own_run(two_runs, filename, marker):
    """The first run's files must not gain the second run's rows, and vice versa."""
    logs_root, _ = two_runs
    for tag, other in (("A", "B"), ("B", "A")):
        path = logs_root / "cells" / tag / filename
        assert path.exists(), f"run {tag} produced no {filename}"
        text = path.read_text()
        assert marker + tag in text, f"{path} is missing its own {marker}{tag}"
        assert marker + other not in text, (
            f"{path} leaked run {other}'s rows -- handlers accumulated across runs"
        )


def test_handler_count_does_not_grow_across_runs(two_runs):
    """Every logger must end each run with the same handler count (3, not 3*N)."""
    _, counts = two_runs
    assert counts["A"] == counts["B"], (
        f"handler counts grew between runs: {counts['A']} -> {counts['B']}"
    )
    assert max(counts["A"].values()) <= 3, counts["A"]


def test_latest_sim_csv_holds_only_the_latest_run(two_runs):
    """logs/latest_sim/*.csv is shared by every run, so it is rewritten each run."""
    logs_root, _ = two_runs
    lines = (logs_root / "latest_sim" / "structured_decisions.csv").read_text().splitlines()
    assert lines[0].startswith("timestamp,round,agent_id"), lines[0]
    assert lines[1:] == ["decision-B"], lines[1:]
