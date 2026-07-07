"""
Aggregate the per-cell outputs of a sweep (see run_sweep.py) into tidy panels.

For each requested CSV (default: structured_decisions.csv and market_data.csv) it
collects that file from every completed cell, prepends the cell's metadata columns
(cell_id, seed, temperature, model, variant, prompt_family, run_dir), and concatenates
everything into a single long / tidy panel written under <sweep_root>/aggregated/.
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


# Metadata columns prepended to every aggregated row, in order. prompt_family is
# the clustering key for inference across runs sharing a prompt family (#102).
META_COLUMNS = ["cell_id", "seed", "temperature", "model", "variant", "prompt_family", "run_dir"]


def cell_prompt_family(cell: dict) -> str:
    """Resolve a cell's prompt-family label, tolerating pre-#102 manifests."""
    return (cell.get("prompt_family")
            or cell.get("param_overrides", {}).get("PROMPT_FAMILY")
            or cell.get("variant")
            or "baseline")

DEFAULT_FILES = ["structured_decisions.csv", "market_data.csv"]


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

    for filename in args.files:
        print(f"\n{filename}:")
        panel = aggregate_file(cells, filename)
        # Name the output after the source stem (e.g. structured_decisions -> decisions_panel).
        stem = Path(filename).stem
        out_path = out_dir / f"{stem}_panel.csv"
        panel.to_csv(out_path, index=False)
        print(f"  -> {len(panel)} rows across {panel['cell_id'].nunique() if len(panel) else 0} cells "
              f"written to {out_path}")

    print(f"\nDone. Panels in {out_dir}")


if __name__ == "__main__":
    main()
