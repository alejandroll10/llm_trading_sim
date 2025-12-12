#!/usr/bin/env python3
"""
Re-run paper scenarios after redemption prompt fix.

Only runs finite-horizon scenarios that were affected by the fix.
Infinite-horizon scenarios (price_discovery_*) are unchanged.

Usage:
    python scripts/rerun_paper_scenarios.py           # Run all affected scenarios
    python scripts/rerun_paper_scenarios.py --dry-run # Show what would run
    python scripts/rerun_paper_scenarios.py --scenario paper_bubble_with_shorts  # Run one
"""

import subprocess
import sys
import argparse
from pathlib import Path

# Scenarios affected by the redemption prompt fix (finite horizon + PROCESS_ONLY)
AFFECTED_SCENARIOS = [
    "paper_social_manipulation",
    "paper_emergent_manipulation",
    "paper_neutral_manipulation",
    "paper_correlated_crash",
    "paper_bubble_with_shorts",
    "paper_bubble_without_shorts",
]

# Scenarios NOT affected (infinite horizon - use different redemption text)
UNAFFECTED_SCENARIOS = [
    "paper_price_discovery_above",
    "paper_price_discovery_below",
]


def run_scenario(scenario_name: str, dry_run: bool = False) -> bool:
    """Run a single scenario using run_base_sim.py"""
    cmd = ["python", "src/run_base_sim.py", scenario_name]

    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True

    print(f"  Running: {scenario_name}...")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"  ✓ {scenario_name} completed successfully")
        return True
    else:
        print(f"  ✗ {scenario_name} FAILED (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Re-run paper scenarios after redemption fix")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without running")
    parser.add_argument("--scenario", type=str, help="Run only this scenario")
    parser.add_argument("--all", action="store_true", help="Run ALL paper scenarios (including unaffected)")
    args = parser.parse_args()

    # Change to repo root
    repo_root = Path(__file__).parent.parent
    import os
    os.chdir(repo_root)

    print("=" * 60)
    print("Re-running paper scenarios after redemption prompt fix")
    print("=" * 60)

    if args.scenario:
        scenarios = [args.scenario]
        print(f"\nRunning single scenario: {args.scenario}")
    elif args.all:
        scenarios = AFFECTED_SCENARIOS + UNAFFECTED_SCENARIOS
        print(f"\nRunning ALL {len(scenarios)} paper scenarios")
    else:
        scenarios = AFFECTED_SCENARIOS
        print(f"\nRunning {len(scenarios)} affected scenarios (finite horizon)")
        print(f"Skipping {len(UNAFFECTED_SCENARIOS)} unaffected scenarios (infinite horizon)")

    print()

    results = {}
    for scenario in scenarios:
        success = run_scenario(scenario, dry_run=args.dry_run)
        results[scenario] = success

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    print(f"Succeeded: {succeeded}/{len(results)}")
    if failed > 0:
        print(f"Failed: {failed}")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
