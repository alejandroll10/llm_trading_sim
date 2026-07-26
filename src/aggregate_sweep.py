"""
Aggregate the per-cell outputs of a sweep (see run_sweep.py) into tidy panels.

For each requested CSV (default: structured_decisions.csv, market_data.csv and
order_data.csv) it collects that file from every completed cell, prepends the cell's
metadata columns (cell_id, seed, temperature, model, variant, prompt_family, run_dir),
and concatenates everything into a single long / tidy panel written under
<sweep_root>/aggregated/.
prompt_family is the clustering key for inference: runs sharing a prompt family
(or model family) are not independent draws.

Usage
-----
    python src/aggregate_sweep.py logs/sweeps/<sweep_name>
    python src/aggregate_sweep.py logs/sweeps/<sweep_name> --files structured_decisions.csv market_data.csv
    python src/aggregate_sweep.py logs/sweeps/<sweep_name> --out-dir /some/where

The sweep root must contain a manifest.json produced by run_sweep.py. Cells whose
status is not "done" are skipped (with a warning). Within each cell's run directory
the CSV is located recursively, so it does not matter whether the file lives at the
run-dir root (structured_decisions.csv) or under data/ (market_data.csv).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd

from services.usage_tracker import aggregate_summaries


# Metadata columns prepended to every aggregated row, in order. prompt_family is
# the clustering key for inference across runs sharing a prompt family (#102).
META_COLUMNS = ["cell_id", "seed", "temperature", "model", "variant", "prompt_family", "run_dir"]


def cell_prompt_family(cell: dict) -> str:
    """Resolve a cell's prompt-family label, tolerating pre-#102 manifests."""
    return (cell.get("prompt_family")
            or cell.get("param_overrides", {}).get("PROMPT_FAMILY")
            or cell.get("variant")
            or "baseline")

# order_data.csv carries every agent's submitted orders (not just the LLM
# agents that write structured_decisions.csv), so it is the only panel that can
# measure market-wide order flow -- the regressor for the realized price-impact
# coefficient in analysis/impact_estimators.py (issue #111 phase 2).
DEFAULT_FILES = ["structured_decisions.csv", "market_data.csv", "order_data.csv"]


# Columns that identify one source row within a cell. If the same key turns up
# under more than one cell_id, the per-cell CSVs overlap -- the signature of log
# rows leaking across runs sharing a process (issue #120) -- and every per-cell
# statistic computed from the panel is contaminated. Note the key need not be
# unique *within* a cell (an agent submitting two orders in one round writes two
# rows with the same timestamp); only its appearance across cells is a problem.
CELL_IDENTITY_KEYS = {
    "structured_decisions.csv": ["timestamp", "agent_id", "round"],
}


def report_cross_cell_duplicates(panel: pd.DataFrame, filename: str) -> int:
    """Warn when one source row is claimed by several cells. Returns #bad keys.

    The timestamp is second-resolution, so two cells that happen to log the same
    agent's decision for the same round within one second would be flagged
    without a real leak. That needs cells short enough to overlap on both round
    and wall-clock second, which no realistic sweep produces -- but it is the
    reason this warns rather than fails the aggregation.
    """
    key = CELL_IDENTITY_KEYS.get(filename)
    if not key or panel.empty or not set(key).issubset(panel.columns):
        return 0

    cells_per_key = panel.groupby(key, dropna=False)["cell_id"].nunique()
    offenders = cells_per_key[cells_per_key > 1]
    if offenders.empty:
        return 0

    examples = ", ".join(
        "(" + ", ".join(str(v) for v in (k if isinstance(k, tuple) else (k,))) + ")"
        for k in offenders.index[:3]
    )
    print(f"  [LEAK] {len(offenders)} of {len(cells_per_key)} distinct "
          f"({'+'.join(key)}) keys appear under more than one cell_id "
          f"(worst: {int(offenders.max())} cells claim the same row).")
    print(f"         e.g. {examples}")
    print(f"         The per-cell {filename} files overlap. This is issue #120: logger "
          f"handlers accumulating across runs in one process, so cell k's file also "
          f"collected later cells' rows. Estimates from this panel are biased -- re-run "
          f"the affected cells with the fix in place before using it.")
    return len(offenders)


def find_csv(run_dir: Path, filename: str):
    """Locate `filename` anywhere under run_dir (root or data/). Returns Path or None."""
    matches = list(run_dir.rglob(filename))
    return matches[0] if matches else None


def aggregate_file(cells, filename: str) -> pd.DataFrame:
    """Build one tidy panel for `filename` across all completed cells."""
    frames = []
    for cell in cells:
        run_dir = Path(cell["run_dir"])
        if not run_dir.exists():
            print(f"  [warn] {cell['cell_id']}: run_dir missing ({run_dir})")
            continue

        csv_path = find_csv(run_dir, filename)
        if csv_path is None:
            print(f"  [warn] {cell['cell_id']}: {filename} not found under {run_dir}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print(f"  [warn] {cell['cell_id']}: {filename} is empty")
            continue

        # Prepend metadata columns.
        meta = {
            "cell_id": cell["cell_id"],
            "seed": cell.get("seed"),
            "temperature": cell.get("temperature"),
            "model": cell.get("model"),
            "variant": cell.get("variant"),
            "prompt_family": cell_prompt_family(cell),
            "run_dir": str(run_dir),
        }
        for col in reversed(META_COLUMNS):
            df.insert(0, col, meta[col])
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=META_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def write_usage_rollup(cells, out_dir: Path) -> None:
    """Roll per-cell realized LLM usage up to a sweep total (issue #104).

    Reads the llm_usage summary each cell carries in the manifest (written by
    run_sweep), writes a per-cell panel (llm_usage_by_cell.csv) and a sweep-total
    summary (llm_usage_summary.json), and prints the headline tokens/$ figure. A
    no-op (with a note) for pre-#104 sweeps whose cells lack usage data.
    """
    summaries = [c.get("llm_usage") for c in cells if c.get("llm_usage")]
    if not summaries:
        print("\nllm_usage: no per-cell usage recorded (pre-#104 sweep?) -- skipping rollup.")
        return

    rollup = aggregate_summaries(summaries)

    # Per-cell panel: one row per cell with its usage totals + cell metadata.
    rows = []
    for cell in cells:
        u = cell.get("llm_usage")
        if not u:
            continue
        rows.append({
            "cell_id": cell["cell_id"],
            "seed": cell.get("seed"),
            "temperature": cell.get("temperature"),
            "model": cell.get("model"),
            "variant": cell.get("variant"),
            "prompt_family": cell_prompt_family(cell),
            "calls": u.get("calls", 0),
            "failed_calls": u.get("failed_calls", 0),
            "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
            "cost_usd": u.get("cost_usd", 0.0),
            "avg_tokens_per_call": u.get("avg_tokens_per_call", 0.0),
        })
    panel_path = out_dir / "llm_usage_by_cell.csv"
    pd.DataFrame(rows).to_csv(panel_path, index=False)

    summary_path = out_dir / "llm_usage_summary.json"
    with open(summary_path, "w") as f:
        json.dump(rollup, f, indent=2)

    print(f"\nllm_usage: {rollup['total_tokens']:,} tokens "
          f"({rollup['prompt_tokens']:,} in / {rollup['completion_tokens']:,} out), "
          f"${rollup['cost_usd']:.4f}, {rollup['calls']:,} calls "
          f"across {rollup['runs_with_usage']} cells")
    if rollup.get("unpriced_models"):
        print(f"  [warn] no price-table entry (counted as $0): "
              f"{', '.join(rollup['unpriced_models'])}")
    print(f"  -> {panel_path}")
    print(f"  -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate sweep cell CSVs into tidy panels with cell metadata columns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("sweep_root", help="Sweep directory containing manifest.json.")
    parser.add_argument("--files", nargs='+', default=DEFAULT_FILES,
                        help=f"CSV filenames to aggregate. Default: {' '.join(DEFAULT_FILES)}")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory. Default: <sweep_root>/aggregated/")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    manifest_path = sweep_root / 'manifest.json'
    if not manifest_path.exists():
        print(f"Error: no manifest.json in {sweep_root}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_cells = list(manifest.get("cells", {}).values())
    cells = [c for c in all_cells if c.get("status") == "done" and c.get("run_dir")]
    skipped = len(all_cells) - len(cells)
    if not cells:
        print(f"Error: no completed cells in {manifest_path} ({len(all_cells)} total).")
        sys.exit(1)
    print(f"Aggregating {len(cells)} completed cells "
          f"({skipped} skipped) from sweep '{manifest.get('sweep_name', sweep_root.name)}'.")

    out_dir = Path(args.out_dir) if args.out_dir else sweep_root / 'aggregated'
    out_dir.mkdir(parents=True, exist_ok=True)

    leaky_files = []
    for filename in args.files:
        print(f"\n{filename}:")
        panel = aggregate_file(cells, filename)
        # Name the output after the source stem (e.g. structured_decisions -> decisions_panel).
        stem = Path(filename).stem
        out_path = out_dir / f"{stem}_panel.csv"
        panel.to_csv(out_path, index=False)
        print(f"  -> {len(panel)} rows across {panel['cell_id'].nunique() if len(panel) else 0} cells "
              f"written to {out_path}")
        if report_cross_cell_duplicates(panel, filename):
            leaky_files.append(filename)

    # Roll up realized token/cost usage across cells (issue #104).
    write_usage_rollup(cells, out_dir)

    print(f"\nDone. Panels in {out_dir}")
    if leaky_files:
        # Repeated at the end so it survives a long scroll-back.
        print(f"\n*** WARNING: cross-cell row leakage detected in "
              f"{', '.join(leaky_files)} (see [LEAK] above). Do not use these "
              f"panels for per-cell inference. ***")


if __name__ == "__main__":
    main()
