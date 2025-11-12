"""Decision analysis visualization functions."""

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from visualization.plot_config import (
    STANDARD_FIGSIZE, WORDCLOUD_FIGSIZE, WORDCLOUD_WIDTH, WORDCLOUD_HEIGHT,
    WORDCLOUD_BACKGROUND, WORDCLOUD_MIN_FONT, GRID_ALPHA
)


def plot_decision_heatmap(decisions_df: pd.DataFrame):
    """
    Plot decision heatmap by agent type (Buy = 1, Sell = -1).

    Args:
        decisions_df: DataFrame with agent decisions

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Convert decision types to binary (Buy = 1, Sell = -1)
    decisions_df = decisions_df.copy()
    decisions_df['decision_value'] = decisions_df['decision'].map({'Buy': 1, 'Sell': -1})

    # Group by round and agent_type, calculate mean decision (-1 to 1)
    decision_heat = decisions_df.groupby(['round', 'agent_type'])['decision_value'].mean().unstack()

    # Plot heatmap
    im = ax.imshow(decision_heat.T, aspect='auto', cmap='RdYlGn',
                   vmin=-1, vmax=1, interpolation='nearest')

    plt.colorbar(im, ax=ax, label='Buy (1) vs Sell (-1)')
    ax.set_xlabel('Round')
    ax.set_ylabel('Agent Type')
    ax.set_title('Agent Decision Patterns Over Time')

    # Set y-axis labels
    ax.set_yticks(range(len(decision_heat.columns)))
    ax.set_yticklabels(decision_heat.columns)

    return fig


def plot_decision_quantities(decisions_df: pd.DataFrame):
    """
    Plot decision quantities by agent type over time.

    Args:
        decisions_df: DataFrame with agent decisions

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    for agent_type in decisions_df['agent_type'].unique():
        agent_mask = decisions_df['agent_type'] == agent_type
        buys = decisions_df[agent_mask & (decisions_df['decision'] == 'Buy')]
        sells = decisions_df[agent_mask & (decisions_df['decision'] == 'Sell')]

        # Plot buys and sells
        if not buys.empty:
            ax.scatter(buys['round'], buys['quantity'],
                      marker='^', label=f'{agent_type} Buys')
        if not sells.empty:
            ax.scatter(sells['round'], -sells['quantity'],
                      marker='v', label=f'{agent_type} Sells')

    ax.set_xlabel('Round')
    ax.set_ylabel('Quantity (negative for sells)')
    ax.set_title('Agent Decision Quantities Over Time')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)

    return fig


def plot_reasoning_wordcloud(reasoning_text: str, title: str):
    """
    Generate a word cloud from reasoning text.

    Args:
        reasoning_text: Concatenated reasoning text
        title: Title for the plot

    Returns:
        Matplotlib figure or None if no text
    """
    if not reasoning_text.strip():
        return None

    fig, ax = plt.subplots(figsize=WORDCLOUD_FIGSIZE)

    # Generate word cloud
    wordcloud = WordCloud(
        width=WORDCLOUD_WIDTH,
        height=WORDCLOUD_HEIGHT,
        background_color=WORDCLOUD_BACKGROUND,
        min_font_size=WORDCLOUD_MIN_FONT
    ).generate(reasoning_text)

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)

    return fig


def generate_all_wordclouds(decisions_df: pd.DataFrame):
    """
    Generate word clouds for all agent types and overall.

    Args:
        decisions_df: DataFrame with agent decisions including reasoning column

    Returns:
        Dict of figures keyed by agent_type (or 'all')
    """
    if 'reasoning' not in decisions_df.columns:
        return {}

    figures = {}

    # Create word cloud for each agent type
    for agent_type in decisions_df['agent_type'].unique():
        agent_text = ' '.join(
            decisions_df[decisions_df['agent_type'] == agent_type]['reasoning']
            .dropna()
            .astype(str)
        )

        fig = plot_reasoning_wordcloud(
            agent_text,
            f'Common Terms in {agent_type} Agent Reasoning'
        )
        if fig is not None:
            figures[agent_type] = fig

    # Create overall word cloud
    all_text = ' '.join(decisions_df['reasoning'].dropna().astype(str))
    fig = plot_reasoning_wordcloud(
        all_text,
        'Common Terms in All Agent Reasoning'
    )
    if fig is not None:
        figures['all'] = fig

    return figures
