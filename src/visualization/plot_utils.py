"""Utility functions for plot generation and data cleaning."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def clean_data(data_list):
    """
    Ensure data is numeric and finite, replacing invalid values with NaN.

    Args:
        data_list: List of values to clean

    Returns:
        List of cleaned numeric values
    """
    if data_list is None:
        return []
    cleaned = []
    for x in data_list:
        try:
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                cleaned.append(np.nan)
            else:
                cleaned.append(float(x))
        except (ValueError, TypeError):
            cleaned.append(np.nan)
    return cleaned


def save_plot(fig, base_name: str, scenario_name: str, dated_plots_dir: Path,
              scenario_plots_dir: Path, close: bool = True):
    """
    Save a plot to both dated and latest scenario directories.

    Args:
        fig: Matplotlib figure to save
        base_name: Base name for the plot file (without extension)
        scenario_name: Name of the scenario (appended to filename)
        dated_plots_dir: Directory for dated run plots
        scenario_plots_dir: Directory for latest scenario plots
        close: Whether to close the figure after saving
    """
    try:
        filename = f'{base_name}_{scenario_name}.png'
        fig.savefig(dated_plots_dir / filename)
        fig.savefig(scenario_plots_dir / filename)
        print(f"  Saved plot: {base_name}")
    except Exception as e:
        print(f"  Error saving plot {base_name}: {str(e)}")
    finally:
        if close:
            plt.close(fig)
