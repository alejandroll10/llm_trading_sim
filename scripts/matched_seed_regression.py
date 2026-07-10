"""Matched-seed regression harness for refactors of the simulation core.

Runs a fixed set of offline scenarios (deterministic agents only, no LLM
calls; RANDOM_SEED is fixed inside run_base_sim) and snapshots each run's
data/ outputs. A later `compare` run re-executes the same scenarios and
diffs every CSV/JSON bit-for-bit, ignoring only wall-clock timestamp
columns. This is the validation approach used for the 2026-07
BaseSimulation decomposition commits (issue #106).

Usage (from the repo root):
    python3 scripts/matched_seed_regression.py capture --out /tmp/baseline
    ... refactor ...
    python3 scripts/matched_seed_regression.py compare --baseline /tmp/baseline

Exit code 0 = all scenarios identical; 1 = any diff or failed run.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Offline scenarios chosen to cover: plain single-stock, multi-stock,
# leverage, short selling (single and multi), intra-round margin calls,
# dividend regime schedules, dividend shocks (systematic/style/mixed),
# and asymmetric information. All use only deterministic agent types.
SCENARIOS = [
    "deterministic_only",
    "multi_basic",
    "single_leverage",
    "single_leverage_short",
    "leverage_stress_test",
    "deterministic_short_selling",
    "multi_short",
    "margin_violation_test",
    "test_regime_shift",
    "systematic_shock_test",
    "style_shock_test",
    "mixed_shock_test",
    "asymmetric_fundamental_regression",
]

# Files under data/ to snapshot and compare. Everything else (plots,
# metadata.json run ids) is timestamp-bearing or derived.
DATA_FILES = [
    "market_data.csv",
    "trade_data.csv",
    "agent_data.csv",
    "order_data.csv",
    "wealth_history.csv",
    "dividend_data.csv",
    "stock_positions.csv",
    "social_messages.csv",
    "summary_statistics.json",
]


def run_scenario(name: str) -> Path:
    """Run one scenario and return its fresh data/ directory."""
    scenario_dir = REPO_ROOT / "logs" / name
    before = set(scenario_dir.glob("*")) if scenario_dir.exists() else set()
    result = subprocess.run(
        [sys.executable, "src/run_base_sim.py", name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Scenario {name} failed (exit {result.returncode}):\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    new_dirs = sorted(set(scenario_dir.glob("*")) - before)
    run_dirs = [d for d in new_dirs if (d / "data").is_dir()]
    if not run_dirs:
        raise RuntimeError(f"Scenario {name}: no new run directory found")
    return run_dirs[-1] / "data"


def snapshot(data_dir: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for fname in DATA_FILES:
        src = data_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)


def normalized_rows(path: Path) -> list:
    """Read a CSV as strings with any 'timestamp' column removed."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return rows
    header = rows[0]
    drop = [i for i, col in enumerate(header) if col == "timestamp"]
    if not drop:
        return rows
    keep = [i for i in range(len(header)) if i not in drop]
    return [[row[i] for i in keep if i < len(row)] for row in rows]


def compare_file(baseline: Path, candidate: Path) -> list:
    """Return a list of human-readable differences (empty = identical)."""
    diffs = []
    if baseline.exists() != candidate.exists():
        missing = "candidate" if baseline.exists() else "baseline"
        return [f"{baseline.name}: missing in {missing}"]
    if not baseline.exists():
        return []
    if baseline.suffix == ".json":
        a = json.loads(baseline.read_text())
        b = json.loads(candidate.read_text())
        if a != b:
            diffs.append(f"{baseline.name}: JSON content differs")
        return diffs
    a_rows = normalized_rows(baseline)
    b_rows = normalized_rows(candidate)
    if len(a_rows) != len(b_rows):
        diffs.append(
            f"{baseline.name}: row count {len(a_rows)} -> {len(b_rows)}"
        )
        return diffs
    for i, (ra, rb) in enumerate(zip(a_rows, b_rows)):
        if ra != rb:
            diffs.append(f"{baseline.name}: first diff at row {i}: {ra} != {rb}")
            break
    return diffs


def capture(out_dir: Path) -> int:
    failures = 0
    for name in SCENARIOS:
        try:
            data_dir = run_scenario(name)
            snapshot(data_dir, out_dir / name)
            print(f"[captured] {name}")
        except RuntimeError as e:
            failures += 1
            print(f"[FAILED]   {name}: {e}")
    return 1 if failures else 0


def compare(baseline_dir: Path, out_dir: Path) -> int:
    failures = 0
    for name in SCENARIOS:
        base = baseline_dir / name
        if not base.is_dir():
            print(f"[SKIP]     {name}: no baseline captured")
            continue
        try:
            data_dir = run_scenario(name)
        except RuntimeError as e:
            failures += 1
            print(f"[FAILED]   {name}: {e}")
            continue
        if out_dir:
            snapshot(data_dir, out_dir / name)
        diffs = []
        for fname in DATA_FILES:
            diffs.extend(compare_file(base / fname, data_dir / fname))
        if diffs:
            failures += 1
            print(f"[DIFF]     {name}")
            for d in diffs:
                print(f"           {d}")
        else:
            print(f"[identical] {name}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_capture = sub.add_parser("capture", help="run scenarios and snapshot outputs")
    p_capture.add_argument("--out", required=True, type=Path)
    p_compare = sub.add_parser("compare", help="re-run scenarios and diff vs baseline")
    p_compare.add_argument("--baseline", required=True, type=Path)
    p_compare.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "capture":
        return capture(args.out)
    return compare(args.baseline, args.out)


if __name__ == "__main__":
    sys.exit(main())
