"""Valuation analysis visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List
from visualization.plot_config import (
    STANDARD_FIGSIZE, EXTRA_LARGE_FIGSIZE,
    FUNDAMENTAL_COLOR, FUNDAMENTAL_LINESTYLE,
    STANDARD_LINEWIDTH, THIN_LINEWIDTH,
    STANDARD_ALPHA, GRID_ALPHA
)


def plot_agent_valuations(decisions_df: pd.DataFrame, history: List[dict],
                          rounds: List[float], price_data: List[float],
                          fundamental_data: List[float], num_trades: List[int] = None):
    """
    Plot agent valuations compared to market price over time.

    Args:
        decisions_df: DataFrame with agent decisions including valuation
        history: List of historical market data
        rounds: List of round numbers
        price_data: List of market prices
        fundamental_data: List of fundamental values
        num_trades: Optional list of trade counts per round (to mark actual trades)

    Returns:
        Matplotlib figure or None if no valuation data
    """
    if 'valuation' not in decisions_df.columns or decisions_df['valuation'].isna().all():
        return None

    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Add actual market price
    ax.plot(rounds, price_data, label='Market Price',
            color='black', linewidth=STANDARD_LINEWIDTH)

    # Mark rounds with actual trades (skip round 1 - initialization)
    if num_trades is not None:
        trade_rounds = [r for r, n in zip(rounds, num_trades) if n and n > 0 and r > 1]
        trade_prices = [p for r, p, n in zip(rounds, price_data, num_trades) if n and n > 0 and r > 1]
        if trade_rounds:
            ax.scatter(trade_rounds, trade_prices, color='black', s=50,
                      zorder=5, label='Actual Trade', marker='o')

    ax.plot(rounds, fundamental_data, label='Fundamental Value',
            color=FUNDAMENTAL_COLOR, linestyle=FUNDAMENTAL_LINESTYLE,
            linewidth=STANDARD_LINEWIDTH)

    # Group valuations by agent type and round
    agent_valuations = decisions_df.groupby(['round', 'agent_type'])['valuation'].mean().unstack()

    # Plot each agent type's valuation
    for agent_type in agent_valuations.columns:
        ax.plot(agent_valuations.index, agent_valuations[agent_type],
               label=f'{agent_type} Valuation',
               linewidth=THIN_LINEWIDTH, alpha=STANDARD_ALPHA)

    ax.set_xlabel('Round')
    ax.set_ylabel('Price / Valuation')
    ax.set_title('Agent Valuations vs Market Price')
    ax.legend(loc='best')
    ax.grid(True, alpha=GRID_ALPHA)

    return fig


def plot_valuation_dispersion(decisions_df: pd.DataFrame, history: List[dict]):
    """
    Plot distribution of agent valuations by type (box plot).

    Args:
        decisions_df: DataFrame with agent decisions including valuation
        history: List of historical market data

    Returns:
        Matplotlib figure or None if no valuation data
    """
    if 'valuation' not in decisions_df.columns or decisions_df['valuation'].isna().all():
        return None

    fig, ax = plt.subplots(figsize=EXTRA_LARGE_FIGSIZE)

    # Sample rounds for clarity
    unique_rounds = sorted(decisions_df['round'].unique())
    sampled_rounds = unique_rounds[::max(1, len(unique_rounds)//10)]  # Sample ~10 rounds

    # Create lists to store handles for legend
    legend_handles = []
    legend_labels = []

    for i, round_num in enumerate(sampled_rounds):
        round_data = decisions_df[decisions_df['round'] == round_num]
        if not round_data.empty:
            data = []
            labels = []
            for agent_type in round_data['agent_type'].unique():
                agent_data = round_data[round_data['agent_type'] == agent_type]['valuation']
                if not agent_data.empty:
                    data.append(agent_data)
                    labels.append(agent_type)

            # Create a boxplot showing the distribution of valuations
            boxplot_positions = [i + j*0.8/len(labels) for j in range(len(labels))]
            bp = ax.boxplot(data, positions=boxplot_positions, widths=0.1,
                           patch_artist=True)

            # Color the boxes according to agent type
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            for j, (patch, color, label) in enumerate(zip(bp['boxes'], colors, labels)):
                patch.set_facecolor(color)

                # Only add to legend if this agent type hasn't been added yet
                if label not in legend_labels:
                    legend_handles.append(patch)
                    legend_labels.append(label)

            # Add price and fundamental as horizontal lines for this round
            if i == 0:
                round_idx = min(round_num, len(history)-1)
                round_price = history[round_idx].get('price', None)
                round_fundamental = history[round_idx].get('fundamental_price', None)

                if round_price is not None:
                    price_line = ax.axhline(y=round_price, color='red', linestyle='-')
                    legend_handles.append(price_line)
                    legend_labels.append('Market Price')

                if round_fundamental is not None:
                    fund_line = ax.axhline(y=round_fundamental, color='green', linestyle='--')
                    legend_handles.append(fund_line)
                    legend_labels.append('Fundamental Value')

    ax.set_xlabel('Round Number')
    ax.set_ylabel('Valuation')
    ax.set_title('Distribution of Agent Valuations by Type')

    # Set x-tick positions at the center of each round's boxplots
    xtick_positions = [i + 0.4 for i in range(len(sampled_rounds))]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(sampled_rounds)

    # Add a legend with all items
    ax.legend(legend_handles, legend_labels, loc='best')
    ax.grid(True, alpha=GRID_ALPHA)

    # Adjust layout to make room for labels
    plt.tight_layout()

    return fig


def plot_price_prediction_accuracy(decisions_df: pd.DataFrame, price_data: List[float],
                                    num_trades: List[int] = None):
    """
    Plot agent price expectations vs realized prices.

    Args:
        decisions_df: DataFrame with agent decisions including price_prediction_t1
        price_data: List of market prices
        num_trades: Optional list of trade counts per round (to mark actual trades)

    Returns:
        Matplotlib figure or None if no price prediction data
    """
    if 'price_prediction_t1' not in decisions_df.columns:
        return None

    fig, ax = plt.subplots(figsize=EXTRA_LARGE_FIGSIZE)

    # Group by round and agent_type
    price_expectations = decisions_df.groupby(['round', 'agent_type'])['price_prediction_t1'].mean().unstack()

    # Calculate next round actual prices
    actual_next_prices = []
    rounds_list = list(price_expectations.index)
    for r in rounds_list:
        if r + 1 < len(price_data):
            actual_next_prices.append(price_data[r + 1])
        else:
            actual_next_prices.append(None)

    # Plot each agent type's price expectation
    for agent_type in price_expectations.columns:
        ax.plot(price_expectations.index, price_expectations[agent_type],
               label=f'{agent_type}',
               linewidth=THIN_LINEWIDTH, alpha=STANDARD_ALPHA)

    # Plot actual next round prices
    ax.plot(rounds_list, actual_next_prices,
           label='Realized Price',
           color='black', linewidth=STANDARD_LINEWIDTH)

    # Mark rounds with actual trades (skip round 1)
    if num_trades is not None and len(num_trades) > 0:
        trade_rounds = [r for r in rounds_list if r + 1 < len(num_trades) and num_trades[r + 1] and num_trades[r + 1] > 0 and r > 0]
        trade_prices = [price_data[r + 1] for r in trade_rounds if r + 1 < len(price_data)]
        if trade_rounds and trade_prices:
            ax.scatter(trade_rounds, trade_prices, color='black', s=50,
                      zorder=5, label='Actual Trade', marker='o')

    ax.set_xlabel('Round')
    ax.set_ylabel('Price')
    ax.set_title('Agent Price Expectations vs Realized Prices')
    ax.legend(loc='best')
    ax.grid(True, alpha=GRID_ALPHA)

    # Make sure there's enough room for labels
    plt.tight_layout()

    return fig


def plot_valuation_vs_expectations(decisions_df: pd.DataFrame, history: List[dict],
                                    rounds: List[float], price_data: List[float],
                                    fundamental_data: List[float], num_trades: List[int] = None):
    """
    Combined plot showing agent valuations and price expectations together.

    This reveals whether agents are value investors (valuation ≈ expectation)
    or speculators (expectation differs from valuation).

    Args:
        decisions_df: DataFrame with agent decisions including valuation and price_prediction_t1
        history: List of historical market data
        rounds: List of round numbers
        price_data: List of market prices
        fundamental_data: List of fundamental values
        num_trades: Optional list of trade counts per round

    Returns:
        Matplotlib figure or None if insufficient data
    """
    has_valuation = 'valuation' in decisions_df.columns and not decisions_df['valuation'].isna().all()
    has_expectations = 'price_prediction_t1' in decisions_df.columns and not decisions_df['price_prediction_t1'].isna().all()

    if not has_valuation and not has_expectations:
        return None

    fig, ax = plt.subplots(figsize=EXTRA_LARGE_FIGSIZE)

    # Plot market price (realized)
    ax.plot(rounds, price_data, label='Market Price',
            color='black', linewidth=STANDARD_LINEWIDTH, zorder=10)

    # Mark rounds with actual trades
    if num_trades is not None:
        trade_rounds = [r for r, n in zip(rounds, num_trades) if n and n > 0 and r > 1]
        trade_prices = [p for r, p, n in zip(rounds, price_data, num_trades) if n and n > 0 and r > 1]
        if trade_rounds:
            ax.scatter(trade_rounds, trade_prices, color='black', s=50,
                      zorder=11, label='Trade Occurred', marker='o')

    # Plot fundamental value
    ax.plot(rounds, fundamental_data, label='Fundamental Value',
            color=FUNDAMENTAL_COLOR, linestyle=FUNDAMENTAL_LINESTYLE,
            linewidth=STANDARD_LINEWIDTH, zorder=9)

    # Get unique agent types
    agent_types = sorted(decisions_df['agent_type'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_types)))

    for agent_type, color in zip(agent_types, colors):
        agent_data = decisions_df[decisions_df['agent_type'] == agent_type]

        # Plot valuation as solid line
        if has_valuation:
            valuations = agent_data.groupby('round')['valuation'].mean()
            ax.plot(valuations.index, valuations.values,
                   label=f'{agent_type} Valuation',
                   color=color, linewidth=THIN_LINEWIDTH, alpha=STANDARD_ALPHA,
                   linestyle='-')

        # Plot price expectation as dashed line
        if has_expectations:
            expectations = agent_data.groupby('round')['price_prediction_t1'].mean()
            ax.plot(expectations.index, expectations.values,
                   label=f'{agent_type} Price Exp.',
                   color=color, linewidth=THIN_LINEWIDTH, alpha=STANDARD_ALPHA,
                   linestyle='--')

    ax.set_xlabel('Round')
    ax.set_ylabel('Price / Valuation')
    ax.set_title('Agent Valuations vs Price Expectations')
    ax.legend(loc='best', fontsize='small', ncol=2)
    ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    return fig


def plot_price_prediction_errors(decisions_df: pd.DataFrame, price_data: List[float]):
    """
    Plot price prediction error distribution by agent type.

    Args:
        decisions_df: DataFrame with agent decisions including price_prediction_t1
        price_data: List of market prices

    Returns:
        Matplotlib figure or None if no price prediction data
    """
    if 'price_prediction_t1' not in decisions_df.columns:
        return None

    fig, ax = plt.subplots(figsize=EXTRA_LARGE_FIGSIZE)

    # Group by round and agent_type
    price_predictions = decisions_df.groupby(['round', 'agent_type'])['price_prediction_t1'].mean().unstack()

    # Calculate next round actual prices
    actual_next_prices = []
    for r in price_predictions.index:
        if r + 1 < len(price_data):
            actual_next_prices.append(price_data[r + 1])
        else:
            actual_next_prices.append(None)

    error_data = []
    agent_types = []

    for agent_type in price_predictions.columns:
        # Calculate errors where we have both predictions and actuals
        errors = []
        for r in price_predictions.index:
            if r < len(actual_next_prices) and actual_next_prices[r] is not None:
                prediction = price_predictions.loc[r, agent_type]
                actual = actual_next_prices[r]
                if not np.isnan(prediction) and not np.isnan(actual):
                    errors.append(abs(prediction - actual) / actual * 100)  # Percent error

        if errors:
            error_data.append(errors)
            agent_types.append(agent_type)

    if error_data:
        ax.boxplot(error_data, labels=agent_types)
        ax.set_ylabel('Absolute Percent Error (%)')
        ax.set_title('Price Prediction Accuracy by Agent Type')
        ax.grid(True, alpha=GRID_ALPHA)
        plt.xticks(rotation=45)

        return fig

    return None
