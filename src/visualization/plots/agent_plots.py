"""Agent-related visualization functions."""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
from visualization.plot_config import (
    STANDARD_FIGSIZE, LARGE_FIGSIZE, WEALTH_COLORS,
    STANDARD_ALPHA, GRID_ALPHA, PER_ROUND_RISK_FREE_RATE
)


def plot_dividend_accumulation(agent_df: pd.DataFrame):
    """
    Plot dividend and interest accumulation by agent type.

    Args:
        agent_df: DataFrame with agent data including dividend_cash column

    Returns:
        Matplotlib figure or None if no dividend data
    """
    if 'dividend_cash' not in agent_df.columns:
        return None

    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Group by round and agent_type, sum for dividend_cash
    grouped = agent_df.groupby(['round', 'agent_type'])['dividend_cash'].sum().unstack()

    # Plot dividend accumulation
    grouped.plot(kind='line', marker='o', ax=ax)

    ax.set_xlabel('Round')
    ax.set_ylabel('Accumulated Dividends & Interest ($)')
    ax.set_title('Dividend & Interest Accumulation by Agent Type')
    ax.legend(title='Agent Type')
    ax.grid(True, alpha=GRID_ALPHA)

    return fig


def plot_wealth_composition_final(agent_df: pd.DataFrame):
    """
    Plot final wealth composition by agent type as stacked bar chart.

    Args:
        agent_df: DataFrame with agent data

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Pick final round for composition
    final_round = agent_df['round'].max()
    final_data = agent_df[agent_df['round'] == final_round]

    # Aggregate by agent type
    agg_dict = {
        'cash': 'sum',
        'share_value': 'sum',
        'total_value': 'sum'
    }
    if 'dividend_cash' in agent_df.columns:
        agg_dict['dividend_cash'] = 'sum'

    wealth_components = final_data.groupby('agent_type').agg(agg_dict)

    # Plot as stacked bar chart
    columns_to_plot = ['cash']
    if 'dividend_cash' in agent_df.columns:
        columns_to_plot.append('dividend_cash')
    columns_to_plot.append('share_value')

    wealth_components[columns_to_plot].plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=WEALTH_COLORS[:len(columns_to_plot)]
    )

    # Add total as text on top of bars
    for i, (agent_type, row) in enumerate(wealth_components.iterrows()):
        total = row['total_value']
        ax.text(i, total + 5, f'${total:.0f}', ha='center', fontweight='bold')

    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Wealth Value ($)')
    ax.set_title(f'Wealth Composition by Agent Type (Round {final_round})')

    labels = ['Trading Cash', 'Dividend & Interest Cash', 'Share Value']
    ax.legend(labels[:len(columns_to_plot)])
    ax.grid(True, alpha=GRID_ALPHA, axis='y')

    return fig


def plot_wealth_composition_overtime(agent_df: pd.DataFrame, agent_type: str):
    """
    Plot wealth composition over time for a specific agent type.

    Args:
        agent_df: DataFrame with agent data
        agent_type: Agent type to plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Filter for this agent type only
    agent_type_data = agent_df[agent_df['agent_type'] == agent_type]

    # Group by round
    agg_dict = {
        'cash': 'sum',
        'share_value': 'sum'
    }
    if 'dividend_cash' in agent_df.columns:
        agg_dict['dividend_cash'] = 'sum'

    by_round = agent_type_data.groupby('round').agg(agg_dict)

    # Create stacked area plot
    columns_to_plot = ['cash']
    if 'dividend_cash' in agent_df.columns:
        columns_to_plot.append('dividend_cash')
    columns_to_plot.append('share_value')

    by_round[columns_to_plot].plot(
        kind='area',
        stacked=True,
        alpha=STANDARD_ALPHA,
        color=WEALTH_COLORS[:len(columns_to_plot)],
        ax=ax
    )

    ax.set_xlabel('Round')
    ax.set_ylabel('Value ($)')
    ax.set_title(f'Wealth Composition Over Time: {agent_type}')

    labels = ['Trading Cash', 'Dividend & Interest Cash', 'Share Value']
    ax.legend(labels[:len(columns_to_plot)])
    ax.grid(True, alpha=GRID_ALPHA)

    return fig


def plot_agent_metric_absolute(agent_df: pd.DataFrame, metric: str, title: str):
    """
    Plot absolute values of an agent metric over time.

    Args:
        agent_df: DataFrame with agent data
        metric: Column name to plot
        title: Plot title

    Returns:
        Matplotlib figure or None if metric not in dataframe
    """
    if metric not in agent_df.columns:
        return None

    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Group by round and agent_type, sum for the metric
    grouped = agent_df.groupby(['round', 'agent_type'])[metric].sum().unstack()

    # Plot absolute values
    grouped.plot(kind='line', marker='o', ax=ax)

    ax.set_xlabel('Round')
    ax.set_ylabel(f'{title}')
    ax.set_title(f'{title} by Agent Type')
    ax.legend(title='Agent Type')
    ax.grid(True, alpha=GRID_ALPHA)

    return fig


def plot_agent_metric_change(agent_df: pd.DataFrame, metric: str, label: str,
                              initial_values: Dict):
    """
    Plot absolute change in metric from initial values.

    Args:
        agent_df: DataFrame with agent data
        metric: Column name to plot
        label: Y-axis label
        initial_values: Dict of initial values by agent type

    Returns:
        Matplotlib figure or None if metric not in dataframe
    """
    if metric not in agent_df.columns:
        return None

    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Group by round and agent_type, sum metric
    grouped = agent_df.groupby(['round', 'agent_type'])[metric].sum().unstack()

    # Calculate absolute change from initial values
    changes = pd.DataFrame(index=grouped.index, columns=grouped.columns)

    for agent_type in grouped.columns:
        if agent_type in initial_values:
            initial = initial_values[agent_type].get(metric, 0)
            changes[agent_type] = grouped[agent_type] - initial

    # Plot changes
    changes.plot(kind='line', marker='o', ax=ax)

    ax.set_xlabel('Round')
    ax.set_ylabel(label)
    ax.set_title(f'{label} by Agent Type')
    ax.legend(title='Agent Type')
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(y=0, color='black', linestyle='-', alpha=GRID_ALPHA)

    return fig


def plot_cash_change(agent_df: pd.DataFrame, initial_values: Dict):
    """
    Plot absolute change in trading cash.

    Args:
        agent_df: DataFrame with agent data
        initial_values: Dict of initial values by agent type

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Group by round and agent_type, sum cash
    grouped = agent_df.groupby(['round', 'agent_type'])['cash'].sum().unstack()

    # Calculate absolute change from initial cash
    cash_changes = pd.DataFrame(index=grouped.index, columns=grouped.columns)

    for agent_type in grouped.columns:
        if agent_type in initial_values:
            initial = initial_values[agent_type]['cash']
            cash_changes[agent_type] = grouped[agent_type] - initial

    # Plot cash changes
    cash_changes.plot(kind='line', marker='o', ax=ax)

    ax.set_xlabel('Round')
    ax.set_ylabel('Change in Trading Cash ($)')
    ax.set_title('Change in Trading Cash Holdings by Agent Type')
    ax.legend(title='Agent Type')
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(y=0, color='black', linestyle='-', alpha=GRID_ALPHA)

    return fig


def plot_cash_returns(agent_df: pd.DataFrame, initial_values: Dict):
    """
    Plot percentage returns on trading cash.

    Args:
        agent_df: DataFrame with agent data
        initial_values: Dict of initial values by agent type

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Group by round and agent_type, sum cash
    grouped = agent_df.groupby(['round', 'agent_type'])['cash'].sum().unstack()

    # Calculate percentage return on cash
    cash_returns = pd.DataFrame(index=grouped.index, columns=grouped.columns)

    for agent_type in grouped.columns:
        if agent_type in initial_values:
            initial = initial_values[agent_type]['cash']
            if initial > 0:
                cash_returns[agent_type] = (grouped[agent_type] / initial - 1) * 100

    # Plot cash returns
    cash_returns.plot(kind='line', marker='o', ax=ax)

    ax.set_xlabel('Round')
    ax.set_ylabel('Change in Trading Cash (%)')
    ax.set_title('Percentage Change in Trading Cash by Agent Type')
    ax.legend(title='Agent Type')
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(y=0, color='black', linestyle='-', alpha=GRID_ALPHA)

    return fig


def plot_wealth_returns(agent_df: pd.DataFrame, initial_values: Dict):
    """
    Plot percentage returns on total wealth.

    Args:
        agent_df: DataFrame with agent data
        initial_values: Dict of initial values by agent type

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Group by round and agent_type, sum total value
    grouped = agent_df.groupby(['round', 'agent_type'])['total_value'].sum().unstack()

    # Calculate percentage return on total value
    value_returns = pd.DataFrame(index=grouped.index, columns=grouped.columns)

    for agent_type in grouped.columns:
        if agent_type in initial_values:
            initial = initial_values[agent_type]['total_value']
            if initial > 0:
                value_returns[agent_type] = (grouped[agent_type] / initial - 1) * 100

    # Plot total value returns
    value_returns.plot(kind='line', marker='o', ax=ax)

    ax.set_xlabel('Round')
    ax.set_ylabel('Change in Total Wealth (%)')
    ax.set_title('Percentage Change on Total Wealth by Agent Type')
    ax.legend(title='Agent Type')
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(y=0, color='black', linestyle='-', alpha=GRID_ALPHA)

    return fig


def plot_excess_returns(agent_df: pd.DataFrame, initial_values: Dict):
    """
    Plot excess returns over risk-free rate.

    Args:
        agent_df: DataFrame with agent data
        initial_values: Dict of initial values by agent type

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Group by round and agent_type, sum total value
    grouped = agent_df.groupby(['round', 'agent_type'])['total_value'].sum().unstack()

    # Calculate percentage returns
    value_returns = pd.DataFrame(index=grouped.index, columns=grouped.columns)
    excess_returns = pd.DataFrame(index=grouped.index, columns=grouped.columns)

    for agent_type in grouped.columns:
        if agent_type in initial_values:
            initial = initial_values[agent_type]['total_value']
            if initial > 0:
                # Raw returns
                value_returns[agent_type] = (grouped[agent_type] / initial - 1) * 100

                # Excess returns (subtracting compounded risk-free rate)
                for idx, round_num in enumerate(value_returns.index):
                    # Account for round numbering
                    t = round_num - min(value_returns.index) if min(value_returns.index) > 0 else round_num

                    # Compound risk-free rate: (1+rf)^t - 1
                    rf_return = ((1 + PER_ROUND_RISK_FREE_RATE) ** t - 1) * 100

                    # Excess return = actual return - risk-free return
                    if not pd.isna(value_returns.at[round_num, agent_type]):
                        excess_returns.at[round_num, agent_type] = (
                            value_returns.at[round_num, agent_type] - rf_return
                        )

    # Plot excess returns
    excess_returns.plot(kind='line', marker='o', ax=ax)

    ax.set_xlabel('Round')
    ax.set_ylabel('Excess Return (%)')
    ax.set_title('Excess Returns Over Risk-Free Rate (5% Per Round)')
    ax.legend(title='Agent Type')
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(y=0, color='black', linestyle='-', alpha=GRID_ALPHA)

    # Add a line showing cumulative risk-free return for reference
    rf_line = [(((1 + PER_ROUND_RISK_FREE_RATE) ** t - 1) * 100)
               for t in range(len(grouped.index))]
    ax.plot(grouped.index, [0] * len(grouped.index), 'k--',
            alpha=0.5, label='Risk-Free Baseline')

    return fig
