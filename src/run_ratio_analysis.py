import json
import numpy as np
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from agents.LLMs.llm_agent import LLMAgent
from agents.LLMs.analysis.signal_generator import MarketScenario
from agents.LLMs.analysis.price_fundamental_analysis import PriceFundamentalAnalyzer

# Convert all warnings to errors
import warnings

warnings.filterwarnings("error")
np.seterr(all="raise")

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis_results"
RATIO_DIR = ANALYSIS_DIR / "ratio_analysis"
LATEST_DIR = RATIO_DIR / "latest"
LOG_DIR = PROJECT_ROOT / "logs"
ARCHIVE_DIR = RATIO_DIR / "archive"

# Analysis parameters
RATIOS = np.linspace(0.1, 3.5, 7)
REPEATS_PER_RATIO = 2

# Agent parameters (matching run_base_sim.py)
INITIAL_CASH = 100000.0
INITIAL_SHARES = 1000
POSITION_LIMIT = 100000000
INITIAL_PRICE = 28.0
MODEL_OPEN_AI = "gpt-4o-2024-11-20"
ALLOW_SHORT_SELLING = False
MARGIN_REQUIREMENT = 0.5
BORROW_RATE = 0.0


def setup_logging(name: str = "ratio_analysis") -> logging.Logger:
    """Setup logging configuration"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{name}.log"
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def setup_test_agent() -> LLMAgent:
    """Initialize a test agent with standard parameters"""
    base_params = {
        "agent_id": "test_agent",
        "initial_cash": INITIAL_CASH,
        "initial_shares": INITIAL_SHARES,
        "position_limit": POSITION_LIMIT,
        "allow_short_selling": ALLOW_SHORT_SELLING,
        "margin_requirement": MARGIN_REQUIREMENT,
        "initial_price": INITIAL_PRICE,
    }

    logger = logging.getLogger("test_agent")
    info_signals_logger = logging.getLogger("test_info_signals")

    return LLMAgent(
        **base_params,
        agent_type="default",
        model_open_ai=MODEL_OPEN_AI,
        logger=logger,
        info_signals_logger=info_signals_logger,
    )


def save_parameters(save_dir: Path):
    """Save analysis parameters"""
    params = {
        "RANDOM_SEED": RANDOM_SEED,
        "RATIOS": RATIOS.tolist(),
        "REPEATS_PER_RATIO": REPEATS_PER_RATIO,
        "INITIAL_CASH": INITIAL_CASH,
        "INITIAL_SHARES": INITIAL_SHARES,
        "POSITION_LIMIT": POSITION_LIMIT,
        "INITIAL_PRICE": INITIAL_PRICE,
        "MODEL_OPEN_AI": MODEL_OPEN_AI,
        "ALLOW_SHORT_SELLING": ALLOW_SHORT_SELLING,
        "MARGIN_REQUIREMENT": MARGIN_REQUIREMENT,
        "BORROW_RATE": BORROW_RATE,
    }

    with open(save_dir / "parameters.json", "w") as f:
        json.dump(params, f, indent=4)


def run_ratio_analysis(
    ratios: List[float] = RATIOS,
    repeats_per_ratio: int = REPEATS_PER_RATIO,
    archive: bool = True,
) -> Dict:
    """Run price/fundamental ratio analysis"""
    logger = setup_logging()
    logger.info(f"Starting ratio analysis with {len(ratios)} ratios")

    # Setup directories
    RATIO_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    save_dir = LATEST_DIR

    if archive:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = ARCHIVE_DIR / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)

        # Copy to latest as well
        if save_dir != LATEST_DIR:
            if LATEST_DIR.exists():
                for file in LATEST_DIR.glob("*"):
                    file.unlink()

    logger.info(f"Results will be saved to: {save_dir}")

    # Create base scenario
    base_scenario = MarketScenario(
        # Core market values
        price=INITIAL_PRICE,
        fundamental_value=INITIAL_PRICE,
        volume=1000.0,
        # Scenario timing
        total_rounds=10,
        current_round=5,
        # Market structure
        spread_percent=0.01,
        order_book_depth=5,
        volume_per_level=100.0,
        # Payment schedules
        dividend_payment_interval=1,
        interest_payment_interval=1,
        dividend_payments_per_year=12,
        # Rates and yields
        interest_rate=0.05,
        dividend_yield=0.05,
        # Keep everything deterministic
        price_history_pattern="constant",
        trade_pattern="alternate",
    )

    # Setup analyzer
    agent = setup_test_agent()
    analyzer = PriceFundamentalAnalyzer(agent, base_scenario, repeats_per_ratio)

    # Open conversation log file
    with open(save_dir / "llm_conversations.txt", "w") as conv_file:

        def log_callback(round_number: int, ratio: float, prompt: str, response: str):
            pbar.update(1)
            conv_file.write(f"\n{'='*80}\n")
            conv_file.write(f"Ratio: {ratio:.2f}, Round {round_number}\n")
            conv_file.write(f"{'='*80}\n\n")
            conv_file.write("PROMPT:\n")
            conv_file.write(prompt)
            conv_file.write("\n\nRESPONSE:\n")
            conv_file.write(response)
            conv_file.write("\n")

        # Run analysis
        with tqdm(
            total=len(ratios) * repeats_per_ratio, desc="Analyzing ratios"
        ) as pbar:
            results = analyzer.run_ratio_analysis(
                ratios=ratios,
                save_dir=save_dir,
                log_callback=log_callback,
                repeats_per_ratio=repeats_per_ratio,
            )

    # Save parameters
    save_parameters(save_dir)

    logger.info("Analysis complete")
    logger.info(f"Results saved to: {save_dir}")

    # Archive handling
    if archive and save_dir != LATEST_DIR:
        for file in save_dir.glob("*"):
            import shutil

            shutil.copy2(file, LATEST_DIR)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run price/fundamental ratio analysis."
    )
    parser.add_argument(
        "--allow-short-selling",
        dest="allow_short_selling",
        action="store_true",
        help="Enable short selling for the test agent.",
    )
    parser.add_argument(
        "--disallow-short-selling",
        dest="allow_short_selling",
        action="store_false",
        help="Disable short selling for the test agent.",
    )
    parser.set_defaults(allow_short_selling=ALLOW_SHORT_SELLING)
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=MARGIN_REQUIREMENT,
        help="Margin requirement for short positions.",
    )
    parser.add_argument(
        "--borrow-rate",
        type=float,
        default=BORROW_RATE,
        help="Borrow rate applied to short positions.",
    )
    args = parser.parse_args()
    ALLOW_SHORT_SELLING = args.allow_short_selling
    MARGIN_REQUIREMENT = args.margin_requirement
    BORROW_RATE = args.borrow_rate

    results = run_ratio_analysis()

    print("\nAnalysis complete!")
    print(f"Results saved to: {results['results_dir']}")

    df = results["dataframe"]
    print("\nDecision distribution by ratio:")
    pivot_table = pd.pivot_table(
        df,
        values="quantity",
        index="current_ratio",
        columns="decision_type",
        aggfunc="count",
        fill_value=0,
    )
    print(pivot_table)

    print("\nAverage quantities by ratio and decision:")
    quantity_table = pd.pivot_table(
        df,
        values="quantity",
        index="current_ratio",
        columns="decision_type",
        aggfunc="mean",
    )
    print(quantity_table)
