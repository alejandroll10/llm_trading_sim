"""Order flow visualization functions."""

import matplotlib.pyplot as plt
import pandas as pd
from visualization.plot_config import STANDARD_FIGSIZE, TALL_FIGSIZE, STANDARD_ALPHA, GRID_ALPHA


def plot_order_flow_by_type(order_df: pd.DataFrame, agent_type_map: dict):
    """
    Plot buy and sell order volumes by agent type (stacked bar charts).

    Args:
        order_df: DataFrame with order data
        agent_type_map: Dict mapping agent_id to agent_type

    Returns:
        Matplotlib figure
    """
    # Add agent types to orders
    order_df = order_df.copy()
    order_df['agent_type'] = order_df['agent_id'].map(agent_type_map)

    # Group orders by round, agent_type and decision
    grouped_orders = order_df.groupby(['round', 'agent_type', 'decision']).agg({
        'quantity': 'sum'
    }).reset_index()

    # Create a plot with two subplots - one for buys, one for sells
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=TALL_FIGSIZE, sharex=True)

    # Process buy orders
    buy_orders = grouped_orders[grouped_orders['decision'] == 'buy']
    if not buy_orders.empty:
        buy_pivot = buy_orders.pivot(index='round', columns='agent_type', values='quantity').fillna(0)
        buy_pivot.plot(kind='bar', stacked=True, ax=ax1, alpha=STANDARD_ALPHA)

    ax1.set_title('Buy Order Volume by Agent Type')
    ax1.set_ylabel('Volume')
    ax1.legend(title='Agent Type')
    ax1.grid(True, alpha=GRID_ALPHA)

    # Process sell orders
    sell_orders = grouped_orders[grouped_orders['decision'] == 'sell']
    if not sell_orders.empty:
        sell_pivot = sell_orders.pivot(index='round', columns='agent_type', values='quantity').fillna(0)
        sell_pivot.plot(kind='bar', stacked=True, ax=ax2, alpha=STANDARD_ALPHA)

    ax2.set_title('Sell Order Volume by Agent Type')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Volume')
    ax2.legend(title='Agent Type')
    ax2.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    return fig


def plot_order_flow_net(order_df: pd.DataFrame, agent_type_map: dict):
    """
    Plot net order flow by agent type (bar chart).

    Args:
        order_df: DataFrame with order data
        agent_type_map: Dict mapping agent_id to agent_type

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Add agent types to orders
    order_df = order_df.copy()
    order_df['agent_type'] = order_df['agent_id'].map(agent_type_map)

    # Group orders by round, agent_type and decision
    grouped_orders = order_df.groupby(['round', 'agent_type', 'decision']).agg({
        'quantity': 'sum'
    }).reset_index()

    # Convert sells to negative for net calculation
    grouped_orders_net = grouped_orders.copy()
    grouped_orders_net.loc[grouped_orders_net['decision'] == 'sell', 'quantity'] *= -1

    # Calculate net order flow
    net_orders = grouped_orders_net.groupby(['round', 'agent_type'])['quantity'].sum().unstack().fillna(0)

    # Plot net order flow
    net_orders.plot(kind='bar', ax=ax)
    ax.axhline(y=0, color='black', linestyle='-', alpha=GRID_ALPHA)
    ax.set_xlabel('Round')
    ax.set_ylabel('Net Order Flow (Positive = Net Buying, Negative = Net Selling)')
    ax.set_title('Net Order Flow by Agent Type')
    ax.legend(title='Agent Type')
    ax.grid(True, alpha=GRID_ALPHA)

    return fig


def plot_order_flow_aggregated(order_df: pd.DataFrame, agent_type_map: dict):
    """
    Plot aggregated order flow by agent type (line chart).

    Args:
        order_df: DataFrame with order data
        agent_type_map: Dict mapping agent_id to agent_type

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Add agent types to orders
    order_df = order_df.copy()
    order_df['agent_type'] = order_df['agent_id'].map(agent_type_map)

    # Group orders by round, agent_type and decision
    grouped_orders = order_df.groupby(['round', 'agent_type', 'decision']).agg({
        'quantity': 'sum'
    }).reset_index()

    # Convert sells to negative for net calculation
    grouped_orders_net = grouped_orders.copy()
    grouped_orders_net.loc[grouped_orders_net['decision'] == 'sell', 'quantity'] *= -1

    # Create a DataFrame with both buys (positive) and sells (negative)
    # Need to sum first because we have multiple rows per (round, agent_type)
    combined_orders = grouped_orders_net.groupby(['round', 'agent_type'])['quantity'].sum().unstack().fillna(0)

    # Plot as lines
    combined_orders.plot(kind='line', marker='o', linewidth=2, ax=ax)

    ax.set_xlabel('Round')
    ax.set_ylabel('Order Volume (Positive = Buy, Negative = Sell)')
    ax.set_title('Net Order Flow by Agent Type')
    ax.legend(title='Agent Type')
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(y=0, color='black', linestyle='-', alpha=GRID_ALPHA)

    return fig
