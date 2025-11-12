"""Trading flow visualization functions."""

import matplotlib.pyplot as plt
import pandas as pd
from visualization.plot_config import STANDARD_FIGSIZE, STANDARD_LINEWIDTH, GRID_ALPHA


def plot_trading_flow(trade_df: pd.DataFrame, agent_type_map: dict):
    """
    Plot trading volume between agent types.

    Args:
        trade_df: DataFrame with trade data
        agent_type_map: Dict mapping agent_id to agent_type

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Add buyer and seller types to trade data
    trade_df = trade_df.copy()
    trade_df['buyer_type'] = trade_df['buyer_id'].map(agent_type_map)
    trade_df['seller_type'] = trade_df['seller_id'].map(agent_type_map)

    # Group by round and calculate volume between agent types
    trade_volume = trade_df.groupby(['round', 'buyer_type', 'seller_type'])['quantity'].sum().reset_index()

    # Get unique agent types
    agent_types = sorted(set(agent_type_map.values()))

    for buyer_type in agent_types:
        for seller_type in agent_types:
            mask = (trade_volume['buyer_type'] == buyer_type) & (trade_volume['seller_type'] == seller_type)
            if mask.any():  # Only plot if there are trades between these types
                ax.plot(trade_volume[mask]['round'],
                       trade_volume[mask]['quantity'],
                       label=f'{seller_type} → {buyer_type}',
                       linewidth=STANDARD_LINEWIDTH)

    ax.set_xlabel('Round')
    ax.set_ylabel('Trading Volume')
    ax.set_title('Trading Volume Between Agent Types')
    ax.legend(title='Trade Direction')
    ax.grid(True, alpha=GRID_ALPHA)

    return fig


def plot_cumulative_trading_flow(trade_df: pd.DataFrame, agent_type_map: dict):
    """
    Plot cumulative net trading flow by agent type.

    Args:
        trade_df: DataFrame with trade data
        agent_type_map: Dict mapping agent_id to agent_type

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Add buyer and seller types to trade data
    trade_df = trade_df.copy()
    trade_df['buyer_type'] = trade_df['buyer_id'].map(agent_type_map)
    trade_df['seller_type'] = trade_df['seller_id'].map(agent_type_map)

    # Get unique agent types
    agent_types = sorted(set(agent_type_map.values()))

    for agent_type in agent_types:
        # Calculate net flow (positive when buying, negative when selling)
        buying_mask = trade_df['buyer_type'] == agent_type
        selling_mask = trade_df['seller_type'] == agent_type

        trade_df_copy = trade_df.copy()
        trade_df_copy['net_flow'] = 0
        trade_df_copy.loc[buying_mask, 'net_flow'] = trade_df_copy.loc[buying_mask, 'quantity']
        trade_df_copy.loc[selling_mask, 'net_flow'] = -trade_df_copy.loc[selling_mask, 'quantity']

        # Calculate cumulative net flow for this agent type
        net_flow = trade_df_copy.groupby('round')['net_flow'].sum()
        cumulative_flow = net_flow.cumsum()

        ax.plot(cumulative_flow.index, cumulative_flow.values,
               label=agent_type,
               linewidth=STANDARD_LINEWIDTH)

    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Net Trading Volume')
    ax.set_title('Cumulative Net Trading Flow by Agent Type')
    ax.legend(title='Agent Type')
    ax.grid(True, alpha=GRID_ALPHA)

    return fig
