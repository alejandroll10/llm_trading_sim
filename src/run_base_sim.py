import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import json
import hashlib
import numpy as np
import random
from base_sim import BaseSimulation
import warnings
from pathlib import Path
from datetime import datetime
from services.logging_service import LoggingService
from scenarios import get_scenario, list_scenarios
import shutil
from visualization.plot_generator import PlotGenerator


def compute_config_hash(parameters: dict) -> str:
    """Compute SHA-256 hash of configuration for reproducibility verification."""
    config_str = json.dumps(parameters, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()


def create_run_directory(sim_type: str, description: str = "", parameters: dict = None) -> Path:
    """Create a directory structure that includes simulation type and date"""
    base_dir = Path('logs')
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create nested structure: logs/sim_type/YYYY-MM-DD_HH-MM-SS/
    run_dir = base_dir / sim_type / date_str
    
    # Add description to metadata rather than folder name to keep paths clean
    metadata = {
        'sim_type': sim_type,
        'description': description,
        'timestamp': date_str,
        'run_id': f"{sim_type}_{date_str}",
        'config_hash': compute_config_hash(parameters) if parameters else None,
        'parameters': parameters
    }
    
    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    with open(run_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return run_dir

def save_parameters(run_dir: Path, params: dict):
    """Save parameters to a JSON file in both run directory and latest_sim"""
    # Save to dated run directory
    with open(run_dir / 'parameters.json', 'w') as f:
        json.dump(params, f, indent=4)
    
    # Save to latest_sim directory with scenario subfolder
    sim_type = run_dir.parts[-2]  # Extract scenario name from run_dir path
    latest_dir = Path('logs') / 'latest_sim'
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    # Create scenario directory if it doesn't exist
    scenario_dir = latest_dir / sim_type
    scenario_dir.mkdir(exist_ok=True)
    
    # Save parameters to scenario directory
    with open(scenario_dir / 'parameters.json', 'w') as f:
        json.dump(params, f, indent=4)

def save_plots(simulation, params: dict):
    """Save all simulation plots to both run directory and latest_sim"""
    plot_generator = PlotGenerator(simulation)
    plot_generator.save_all_plots()

def copy_data_to_latest(simulation):
    """Copy all relevant data files to latest_sim directory, organized by scenario"""
    latest_dir = Path('logs') / 'latest_sim'
    
    # Create latest_sim directory if it doesn't exist
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    # Debug print
    print(f"Copying data to latest_sim for scenario: {simulation.sim_type}")
    
    # Create a scenario-specific subdirectory
    scenario_dir = latest_dir / simulation.sim_type
    print(f"Scenario directory path: {scenario_dir}")
    
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # First check if we have source data to copy
    source_data_dir = simulation.run_dir / 'data'
    source_plots_dir = simulation.run_dir / 'plots'
    metadata_file = simulation.run_dir / 'metadata.json'
    params_file = simulation.run_dir / 'parameters.json'
    
    print(f"Source data directory: {source_data_dir} (exists: {source_data_dir.exists()})")
    print(f"Source plots directory: {source_plots_dir} (exists: {source_plots_dir.exists()})")
    
    # Handle data directory - copy files individually without removing directory
    if source_data_dir.exists() and any(source_data_dir.iterdir()):
        scenario_data_dir = scenario_dir / 'data'
        scenario_data_dir.mkdir(exist_ok=True)
        print(f"Created scenario data directory: {scenario_data_dir}")
        
        # Copy each file individually
        for file in source_data_dir.glob('*'):
            target_file = scenario_data_dir / file.name
            shutil.copy2(file, target_file)
            print(f"Copied {file} to {target_file}")
    
    # Handle plots directory - copy files individually without removing directory
    if source_plots_dir.exists() and any(source_plots_dir.iterdir()):
        scenario_plots_dir = scenario_dir / 'plots'
        scenario_plots_dir.mkdir(exist_ok=True)
        print(f"Created scenario plots directory: {scenario_plots_dir}")
        
        # Copy each file individually
        for file in source_plots_dir.glob('*'):
            target_file = scenario_plots_dir / file.name
            shutil.copy2(file, target_file)
            print(f"Copied {file} to {target_file}")
    
    # Copy metadata.json if it exists
    if metadata_file.exists():
        target_metadata = scenario_dir / 'metadata.json'
        shutil.copy2(metadata_file, target_metadata)
        print(f"Copied metadata to {target_metadata}")
    
    # Copy parameters.json if it exists
    if params_file.exists():
        target_params = scenario_dir / 'parameters.json'
        shutil.copy2(params_file, target_params)
        print(f"Copied parameters to {target_params}")

def run_scenario(
    scenario_name: str,
    allow_short_selling: bool = None,
    margin_requirement: float = None,
    borrow_rate: float = None,
):
    """Run a single scenario by name"""
    # Load scenario
    scenario = get_scenario(scenario_name)
    params = scenario.parameters

    # Apply overrides if provided
    agent_params = params.get("AGENT_PARAMS", {})
    if allow_short_selling is not None:
        agent_params['allow_short_selling'] = allow_short_selling
    if margin_requirement is not None:
        agent_params['margin_requirement'] = margin_requirement
    if borrow_rate is not None:
        borrow_model = agent_params.get('borrow_model', {})
        borrow_model['rate'] = borrow_rate
        borrow_model.setdefault('payment_frequency', 1)
        agent_params['borrow_model'] = borrow_model

    # Set random seeds for reproducibility
    np.random.seed(params["RANDOM_SEED"])
    random.seed(params["RANDOM_SEED"])

    # Create run directory with scenario info
    run_dir = create_run_directory(
        sim_type=scenario.name,
        description=scenario.description,
        parameters=params
    )

    # Check if this is a multi-stock scenario
    is_multi_stock = params.get("IS_MULTI_STOCK", False)

    if is_multi_stock:
        # Multi-stock scenario: pass stock_configs instead of single stock params
        simulation = BaseSimulation(
            num_rounds=params["NUM_ROUNDS"],
            initial_price=0,  # Unused for multi-stock, but required parameter
            fundamental_price=0,  # Unused for multi-stock, but required parameter
            redemption_value=None,
            transaction_cost=params.get("TRANSACTION_COST", 0.0),
            lendable_shares=params.get("LENDABLE_SHARES", 0),
            agent_params=params["AGENT_PARAMS"],
            dividend_params=None,  # Per-stock dividend params in stock_configs
            model_open_ai=params["MODEL_OPEN_AI"],
            interest_params=params["INTEREST_MODEL"],
            enable_intra_round_margin_checking=params.get("ENABLE_INTRA_ROUND_MARGIN_CHECKING", False),
            fundamental_info_mode=params["FUNDAMENTAL_INFO_MODE"],
            infinite_rounds=params["INFINITE_ROUNDS"],
            sim_type=scenario.name,
            stock_configs=params["STOCKS"],  # NEW: Pass stock configurations
            news_enabled=params.get("NEWS_ENABLED", False)
        )
    else:
        # Single-stock scenario: original behavior (backwards compatible)
        redemption_value = params.get("REDEMPTION_VALUE", None)

        simulation = BaseSimulation(
            num_rounds=params["NUM_ROUNDS"],
            initial_price=params["INITIAL_PRICE"],
            fundamental_price=params["FUNDAMENTAL_PRICE"],
            redemption_value=redemption_value,
            transaction_cost=params["TRANSACTION_COST"],
            lendable_shares=params.get("LENDABLE_SHARES", 0),
            agent_params=params["AGENT_PARAMS"],
            dividend_params=params["DIVIDEND_PARAMS"],
            model_open_ai=params["MODEL_OPEN_AI"],
            interest_params=params["INTEREST_MODEL"],
            fundamental_info_mode=params["FUNDAMENTAL_INFO_MODE"],
            infinite_rounds=params["INFINITE_ROUNDS"],
            sim_type=scenario.name,
            enable_intra_round_margin_checking=params.get("ENABLE_INTRA_ROUND_MARGIN_CHECKING", False),
            news_enabled=params.get("NEWS_ENABLED", False)
        )

    # Save parameters and run simulation
    save_parameters(simulation.run_dir, params)
    simulation.run()
    save_plots(simulation, params)
    
    # Copy all data files to latest_sim
    copy_data_to_latest(simulation)

    # Print final agent states
    for agent_id in simulation.agent_repository.get_all_agent_ids():
        # Pass prices dict for multi-stock or single price for single-stock
        if simulation.is_multi_stock:
            prices = {stock_id: context.current_price for stock_id, context in simulation.contexts.items()}
        else:
            prices = simulation.context.current_price

        state = simulation.agent_repository.get_agent_state_snapshot(
            agent_id,
            prices
        )
        # Use wealth from snapshot (correctly calculated for both single and multi-stock)
        print(f"Agent {state.agent_id} Type: {state.agent_type} - "
            f"Cash: ${state.cash:.2f}, "
            f"Shares: {state.total_shares}, "
            f"Total Value: ${state.wealth:.2f}")

def main():
    """
    Main function to run simulations.
    Parses command-line arguments to run a specific scenario or list available ones.
    """
    import argparse
    
    # Don't convert warnings to errors - allow normal warnings
    np.seterr(all='warn')

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a trading simulation scenario.")
    parser.add_argument(
        "scenario", 
        nargs='?', 
        default=None, 
        help="The name of the scenario to run. If not provided, lists available scenarios."
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available scenarios and their descriptions."
    )
    parser.add_argument(
        "--allow-short-selling",
        dest="allow_short_selling",
        action="store_true",
        help="Enable short selling regardless of scenario settings"
    )
    parser.add_argument(
        "--disallow-short-selling",
        dest="allow_short_selling",
        action="store_false",
        help="Disable short selling regardless of scenario settings"
    )
    parser.set_defaults(allow_short_selling=None)
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=None,
        help="Override margin requirement for agents"
    )
    parser.add_argument(
        "--borrow-rate",
        type=float,
        default=None,
        help="Override borrow rate for short positions"
    )

    args = parser.parse_args()

    # Get available scenarios
    available_scenarios = list_scenarios()

    # If --list is used, print scenarios and exit
    if args.list:
        print("Available scenarios:")
        for name, desc in available_scenarios.items():
            print(f"  - {name}: {desc}")
        return

    # If no scenario is provided, print list and exit
    if args.scenario is None:
        print("No scenario specified. Please choose from the list below:")
        for name, desc in available_scenarios.items():
            print(f"  - {name}")
        print("\nUsage: python src/run_base_sim.py <scenario_name>")
        return

    # Check if the chosen scenario exists
    if args.scenario not in available_scenarios:
        print(f"Error: Scenario '{args.scenario}' not found.")
        print("Please choose from the available scenarios:")
        for name in available_scenarios.keys():
            print(f"  - {name}")
        return
        
    # Run the selected scenario
    scenario_name = args.scenario
    print(f"\nRunning scenario: {scenario_name}")
    print("-" * 50)
    try:
        run_scenario(
            scenario_name,
            allow_short_selling=args.allow_short_selling,
            margin_requirement=args.margin_requirement,
            borrow_rate=args.borrow_rate,
        )
        print(f"Successfully completed scenario: {scenario_name}")
    except Exception as e:
        print(f"Error running scenario {scenario_name}: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()