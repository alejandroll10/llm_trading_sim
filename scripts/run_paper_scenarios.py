#!/usr/bin/env python3
"""
Run Paper Scenarios - Reproducibility Script

This script runs all scenarios defined for the Management Science paper
and saves results to a versioned directory with full provenance tracking.

Usage:
    python scripts/run_paper_scenarios.py              # Run all paper scenarios
    python scripts/run_paper_scenarios.py --list       # List scenarios without running
    python scripts/run_paper_scenarios.py --dry-run    # Show what would be run
    python scripts/run_paper_scenarios.py scenario1 scenario2  # Run specific scenarios

Output:
    logs/paper_v1_YYYYMMDD_HHMMSS/
    ├── manifest.json          # Provenance: git commit, timestamps, parameters
    ├── paper_no_trade_homogeneous/
    │   ├── data/
    │   ├── plots/
    │   ├── metadata.json
    │   └── parameters.json
    ├── paper_no_trade_heterogeneous/
    │   └── ...
    └── ...
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scenarios.paper_management_science import PAPER_SCENARIO_NAMES, PAPER_VERSION


def get_git_info():
    """Get current git commit and status for provenance tracking."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Check if working directory is clean
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        is_clean = len(status) == 0

        return {
            'commit': commit,
            'branch': branch,
            'is_clean': is_clean,
            'dirty_files': status.split('\n') if status else []
        }
    except subprocess.CalledProcessError:
        return {'commit': 'unknown', 'branch': 'unknown', 'is_clean': False}


def create_paper_output_dir():
    """Create versioned output directory for paper results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f"paper_{PAPER_VERSION}_{timestamp}"
    output_dir = Path('logs') / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_manifest(output_dir: Path, scenarios_to_run: list, start_time: datetime):
    """Create manifest.json with full provenance information."""
    git_info = get_git_info()

    manifest = {
        'paper_version': PAPER_VERSION,
        'created_at': start_time.isoformat(),
        'git': git_info,
        'python_version': sys.version,
        'scenarios': scenarios_to_run,
        'output_directory': str(output_dir),
        'status': 'in_progress',
        'results': {}
    }

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def update_manifest(manifest_path: Path, scenario_name: str, status: str, duration: float = None):
    """Update manifest with scenario completion status."""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    manifest['results'][scenario_name] = {
        'status': status,
        'completed_at': datetime.now().isoformat(),
        'duration_seconds': duration
    }

    # Check if all scenarios are done
    if len(manifest['results']) == len(manifest['scenarios']):
        all_success = all(r['status'] == 'success' for r in manifest['results'].values())
        manifest['status'] = 'completed' if all_success else 'completed_with_errors'
        manifest['completed_at'] = datetime.now().isoformat()

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def run_scenario(scenario_name: str, output_dir: Path):
    """Run a single scenario and save to the paper output directory."""
    import time
    from base_sim import BaseSimulation
    from scenarios import get_scenario
    from visualization.plot_generator import PlotGenerator
    import shutil

    print(f"\n{'='*60}")
    print(f"Running: {scenario_name}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Get scenario
        scenario = get_scenario(scenario_name)
        params = scenario.parameters

        # Create scenario output directory
        scenario_dir = output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Run simulation
        sim = BaseSimulation(
            sim_type=scenario_name,
            description=scenario.description,
            params=params
        )
        sim.run()

        # Generate plots
        plot_generator = PlotGenerator(sim)
        plot_generator.save_all_plots()

        # Copy all results to paper directory (use copytree to get everything)
        source_dir = sim.run_dir

        # Copy key directories and files
        for item in ['data', 'plots']:
            src_path = source_dir / item
            if src_path.exists():
                shutil.copytree(src_path, scenario_dir / item, dirs_exist_ok=True)

        # Copy all important files at root level
        for fname in ['metadata.json', 'parameters.json', 'structured_decisions.csv',
                      'margin_calls.csv', 'validation_errors.csv']:
            if (source_dir / fname).exists():
                shutil.copy2(source_dir / fname, scenario_dir / fname)

        duration = time.time() - start_time
        print(f"✓ Completed {scenario_name} in {duration:.1f}s")
        return 'success', duration

    except Exception as e:
        duration = time.time() - start_time
        print(f"✗ Failed {scenario_name}: {e}")
        return 'failed', duration


def main():
    parser = argparse.ArgumentParser(
        description='Run paper scenarios for reproducibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('scenarios', nargs='*', help='Specific scenarios to run (default: all)')
    parser.add_argument('--list', action='store_true', help='List available scenarios')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without running')

    args = parser.parse_args()

    # List mode
    if args.list:
        print(f"\nPaper Scenarios (version {PAPER_VERSION}):")
        print("-" * 50)
        for name in PAPER_SCENARIO_NAMES:
            print(f"  • {name}")
        print(f"\nTotal: {len(PAPER_SCENARIO_NAMES)} scenarios")
        return

    # Determine which scenarios to run
    if args.scenarios:
        scenarios_to_run = []
        for s in args.scenarios:
            if s in PAPER_SCENARIO_NAMES:
                scenarios_to_run.append(s)
            elif f"paper_{s}" in PAPER_SCENARIO_NAMES:
                scenarios_to_run.append(f"paper_{s}")
            else:
                print(f"Warning: Unknown scenario '{s}', skipping")
        if not scenarios_to_run:
            print("No valid scenarios specified")
            return
    else:
        scenarios_to_run = PAPER_SCENARIO_NAMES

    # Dry run mode
    if args.dry_run:
        print(f"\nDry run - would run {len(scenarios_to_run)} scenarios:")
        for name in scenarios_to_run:
            print(f"  • {name}")
        return

    # Check git status
    git_info = get_git_info()
    if not git_info['is_clean']:
        print("\n⚠️  WARNING: Git working directory is not clean!")
        print("   For full reproducibility, commit your changes first.")
        print(f"   Dirty files: {git_info['dirty_files'][:5]}...")
        response = input("   Continue anyway? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Create output directory
    start_time = datetime.now()
    output_dir = create_paper_output_dir()
    print(f"\n📁 Output directory: {output_dir}")

    # Create manifest
    manifest_path = create_manifest(output_dir, scenarios_to_run, start_time)
    print(f"📋 Manifest: {manifest_path}")
    print(f"🔗 Git commit: {git_info['commit'][:8]}")

    # Run scenarios
    print(f"\n🚀 Running {len(scenarios_to_run)} scenarios...")

    # Change to src directory for imports
    os.chdir(Path(__file__).parent.parent / 'src')

    results = []
    for i, scenario_name in enumerate(scenarios_to_run, 1):
        print(f"\n[{i}/{len(scenarios_to_run)}]", end="")
        status, duration = run_scenario(scenario_name, output_dir)
        update_manifest(manifest_path, scenario_name, status, duration)
        results.append((scenario_name, status, duration))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    success = sum(1 for _, s, _ in results if s == 'success')
    failed = sum(1 for _, s, _ in results if s == 'failed')
    total_time = sum(d for _, _, d in results)

    print(f"✓ Success: {success}/{len(results)}")
    if failed > 0:
        print(f"✗ Failed: {failed}/{len(results)}")
    print(f"⏱ Total time: {total_time/60:.1f} minutes")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📋 Manifest: {manifest_path}")

    # Create symlink to latest paper run
    latest_link = Path('logs') / f'paper_{PAPER_VERSION}_latest'
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.name)
    print(f"🔗 Symlink: {latest_link} -> {output_dir.name}")


if __name__ == '__main__':
    main()
