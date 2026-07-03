"""
Multi-run sweep driver.

Runs a single base scenario across a grid of

    seeds  x  temperatures  x  models  ( x  prompt/param variants )

writing one output directory per cell plus a resumable manifest, and printing a
cost estimate (approximate LLM API call count) before launching.

This is the workhorse for robustness / distribution reporting: nonzero
temperature also removes the mechanical no-trade that identical zero-temperature
agents force, which is needed for the asymmetric-info No-Trade experiments.

Output layout (under logs/):

    logs/sweeps/<sweep_name>/
        manifest.json                     # sweep config + per-cell status (resumable)
        <cell_id>/<timestamp>/            # a normal simulation run directory
            data/*.csv
            plots/*.png
            parameters.json
            metadata.json

Each cell sets both RANDOM_SEED (Python-side RNG: dividends, ordering) and
LLM_SEED (the API sampling seed) to the cell's seed value, so a "seed" is a full
re-randomization of the run. Use --separate-seeds to hold RANDOM_SEED fixed and
vary only the LLM sampling seed.

Prompt / param variants
-----------------------
A "variant" is a named bundle of extra top-level parameter overrides. This is how
prompt variants (or any other axis) are expressed. Define them in a JSON file
passed with --variants-file:

    {
      "baseline":     {},
      "hidden_fv":    {"FUNDAMENTAL_INFO_MODE": "realizations_only"},
      "with_news":    {"NEWS_ENABLED": true}
    }

If no variants file is given, a single "baseline" variant (no extra overrides) is used.

Overrides are applied as a shallow top-level merge onto the scenario parameters, so
to tweak a nested key (e.g. inside AGENT_PARAMS) a variant must supply the whole
top-level dict for that key.

Examples
--------
    # Dry run: just print the grid and cost estimate, run nothing
    python src/run_sweep.py simple_mixed_traders \
        --seeds 42 7 13 --temperatures 0.0 0.5 1.0 --dry-run

    # Real sweep across two models, auto-confirm
    python src/run_sweep.py simple_mixed_traders \
        --seeds 42 7 --temperatures 0.0 1.0 \
        --models gpt-oss-120b gpt-oss-20b --yes

    # Resume an interrupted sweep (skips cells already completed in the manifest)
    python src/run_sweep.py simple_mixed_traders --seeds 42 7 --temperatures 0.0 1.0 \
        --sweep-name my_sweep --resume --yes
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (must be set before pyplot import)

import os
import sys
import copy
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

from scenarios import get_scenario
from scenarios.base import DEFAULT_LLM_MODEL
from run_base_sim import run_scenario


def sanitize(value) -> str:
    """Make a value safe to embed in a directory name."""
    s = str(value)
    for ch in ('/', '\\', ' ', ':', '='):
        s = s.replace(ch, '-')
    return s


def cell_id(seed, temperature, model, variant_name) -> str:
    """Deterministic, human-readable id for a sweep cell."""
    return f"s{sanitize(seed)}_t{sanitize(temperature)}_m{sanitize(model)}_v{sanitize(variant_name)}"


def count_llm_agents(params: dict) -> int:
    """Number of agents that make an LLM API call each round.

    Excludes deterministic (rule-based) agents and the hold_llm agent, neither of
    which hits the API.
    """
    from base_sim import DETERMINISTIC_AGENTS  # imported here to avoid heavy import at module load

    composition = params.get("AGENT_PARAMS", {}).get("agent_composition", {})
    total = 0
    for agent_type, count in composition.items():
        if agent_type in DETERMINISTIC_AGENTS:
            continue
        if agent_type == "hold_llm":
            continue
        total += count
    return total


def estimate_calls_per_cell(base_params: dict, overrides: dict = None) -> int:
    """Approximate LLM API calls for a single cell (one full simulation run).

    `overrides` (a cell's variant/param overrides) is merged onto a copy of
    base_params first, so variants that flip NEWS_ENABLED or change NUM_ROUNDS /
    agent_composition are costed accurately rather than under-reported.
    """
    params = copy.deepcopy(base_params)
    if overrides:
        params.update(overrides)

    num_rounds = params.get("NUM_ROUNDS", 0)
    llm_agents = count_llm_agents(params)
    calls = num_rounds * llm_agents
    # News generation makes its own extra calls (roughly one per round when enabled).
    if params.get("NEWS_ENABLED", False):
        calls += num_rounds
    return calls


def build_cells(seeds, temperatures, models, variants):
    """Cartesian product of the grid axes → ordered list of cell specs."""
    cells = []
    for variant_name, variant_overrides in variants.items():
        for model in models:
            for temperature in temperatures:
                for seed in seeds:
                    cells.append({
                        "cell_id": cell_id(seed, temperature, model, variant_name),
                        "seed": seed,
                        "temperature": temperature,
                        "model": model,
                        "variant": variant_name,
                        "variant_overrides": variant_overrides,
                    })
    return cells


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def save_manifest(manifest_path: Path, manifest: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Run a scenario across a grid of seeds x temperatures x models x variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("scenario", help="Base scenario name to sweep.")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42],
                        help="Seed values (each sets RANDOM_SEED and LLM_SEED). Default: 42")
    parser.add_argument("--temperatures", type=float, nargs='+', default=[0.0],
                        help="Sampling temperatures. Default: 0.0")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                        help="Model names. Default: the scenario's MODEL_OPEN_AI.")
    parser.add_argument("--variants-file", type=str, default=None,
                        help="JSON file mapping variant name -> dict of extra param overrides.")
    parser.add_argument("--separate-seeds", action="store_true",
                        help="Vary only LLM_SEED; hold RANDOM_SEED at the scenario default.")
    parser.add_argument("--sweep-name", type=str, default=None,
                        help="Name for the sweep output dir. Default: <scenario>_<timestamp>.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells already marked 'done' in an existing manifest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the grid and cost estimate, then exit without running.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive confirmation prompt.")
    args = parser.parse_args()

    # Validate scenario up front and grab its parameters for cost estimation.
    try:
        scenario = get_scenario(args.scenario)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    base_params = scenario.parameters

    # Resolve axes.
    models = args.models if args.models else [base_params.get("MODEL_OPEN_AI", DEFAULT_LLM_MODEL)]

    if args.variants_file:
        with open(args.variants_file) as f:
            variants = json.load(f)
        if not isinstance(variants, dict) or not variants:
            print("Error: --variants-file must contain a non-empty JSON object.")
            sys.exit(1)
    else:
        variants = {"baseline": {}}

    cells = build_cells(args.seeds, args.temperatures, models, variants)

    # Sweep output root + manifest.
    if args.resume and not args.sweep_name:
        print("Warning: --resume without --sweep-name has no effect (a fresh timestamped "
              "sweep name is generated, so no prior manifest can be found). "
              "Pass --sweep-name <name> to resume an existing sweep.")
    sweep_name = args.sweep_name or f"{args.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_root = Path('logs') / 'sweeps' / sweep_name
    manifest_path = sweep_root / 'manifest.json'

    manifest = load_manifest(manifest_path) if args.resume else {}
    manifest.setdefault("sweep_name", sweep_name)
    manifest.setdefault("base_scenario", args.scenario)
    manifest.setdefault("created", datetime.now().isoformat())
    manifest["axes"] = {
        "seeds": args.seeds,
        "temperatures": args.temperatures,
        "models": models,
        "variants": list(variants.keys()),
        "separate_seeds": args.separate_seeds,
    }
    manifest.setdefault("cells", {})

    # Cost estimate (per-cell so variant overrides that change call count are honored).
    done_ids = {cid for cid, c in manifest["cells"].items() if c.get("status") == "done"}
    remaining_cells = [c for c in cells if not (args.resume and c["cell_id"] in done_ids)]
    for c in cells:
        c["est_calls"] = estimate_calls_per_cell(base_params, c["variant_overrides"])
    total_calls = sum(c["est_calls"] for c in remaining_cells)

    print(f"\n=== Sweep: {sweep_name} ===")
    print(f"Base scenario : {args.scenario}")
    print(f"Grid          : {len(args.seeds)} seeds x {len(args.temperatures)} temps "
          f"x {len(models)} models x {len(variants)} variants = {len(cells)} cells")
    if args.resume and done_ids:
        print(f"Resuming      : {len(done_ids)} already done, {len(remaining_cells)} to run")
    print(f"Est. LLM calls: ~{total_calls} total across {len(remaining_cells)} cells")
    print(f"Output root   : {sweep_root}\n")
    print("Cells to run:")
    for c in remaining_cells:
        print(f"  - {c['cell_id']}  (seed={c['seed']}, temp={c['temperature']}, "
              f"model={c['model']}, variant={c['variant']}, ~{c['est_calls']} calls)")
    print()

    if args.dry_run:
        print("Dry run: nothing executed.")
        # Still persist the planned manifest so --resume has something to read.
        save_manifest(manifest_path, manifest)
        return

    if not remaining_cells:
        print("Nothing to run (all cells already done).")
        save_manifest(manifest_path, manifest)
        return

    if not args.yes:
        reply = input(f"Launch {len(remaining_cells)} runs (~{total_calls} LLM calls)? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return

    save_manifest(manifest_path, manifest)

    # Run cells sequentially (ONE simulation at a time — see CLAUDE.md rate-limit note).
    for i, c in enumerate(remaining_cells, 1):
        cid = c["cell_id"]
        print(f"\n[{i}/{len(remaining_cells)}] Running cell {cid} ...")

        param_overrides = {
            "LLM_TEMPERATURE": c["temperature"],
            "LLM_SEED": c["seed"],
            "MODEL_OPEN_AI": c["model"],
        }
        if not args.separate_seeds:
            param_overrides["RANDOM_SEED"] = c["seed"]
        # Variant overrides win over the axis overrides above.
        param_overrides.update(c["variant_overrides"])

        # Nest each cell under the sweep root: logs/sweeps/<name>/<cell_id>/<timestamp>/
        sim_type_override = f"sweeps/{sweep_name}/{cid}"

        cell_record = {
            "cell_id": cid,
            "seed": c["seed"],
            "temperature": c["temperature"],
            "model": c["model"],
            "variant": c["variant"],
            "param_overrides": param_overrides,
            "sim_type": sim_type_override,
        }

        try:
            simulation = run_scenario(
                args.scenario,
                param_overrides=param_overrides,
                sim_type_override=sim_type_override,
            )
            cell_record["status"] = "done"
            cell_record["run_dir"] = str(simulation.run_dir)
            print(f"  -> done: {simulation.run_dir}")
        except Exception as e:
            cell_record["status"] = "failed"
            cell_record["error"] = str(e)
            print(f"  -> FAILED: {e}")
            traceback.print_exc()

        # Persist after every cell so the sweep is resumable even if interrupted.
        manifest["cells"][cid] = cell_record
        save_manifest(manifest_path, manifest)

    done = sum(1 for c in manifest["cells"].values() if c.get("status") == "done")
    failed = sum(1 for c in manifest["cells"].values() if c.get("status") == "failed")
    print(f"\n=== Sweep complete: {done} done, {failed} failed ===")
    print(f"Manifest: {manifest_path}")
    print(f"Aggregate with: python src/aggregate_sweep.py {sweep_root}")


if __name__ == "__main__":
    main()
