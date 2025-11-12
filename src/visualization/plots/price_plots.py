"""Price-related visualization functions."""

import matplotlib.pyplot as plt
from typing import List
from visualization.plot_config import (
    STANDARD_FIGSIZE, FUNDAMENTAL_COLOR, PRICE_COLOR, MIDPOINT_COLOR,
    BID_COLOR, ASK_COLOR, SPREAD_COLOR, SHORT_EXPOSURE_COLOR,
    FUNDAMENTAL_LINESTYLE, STANDARD_LINEWIDTH, STANDARD_ALPHA, MEDIUM_ALPHA,
    SPREAD_ALPHA, GRID_ALPHA
)


def plot_price_vs_fundamental(rounds: List[float], fundamental_prices: List[float],
                               last_trade_prices: List[float], midpoint_prices: List[float]):
    """
    Plot market price evolution compared to fundamental value.

    Args:
        rounds: List of round numbers
        fundamental_prices: List of fundamental values
        last_trade_prices: List of last trade prices
        midpoint_prices: List of midpoint prices

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    ax.plot(rounds, fundamental_prices, label='Fundamental Value',
            linestyle=FUNDAMENTAL_LINESTYLE, linewidth=STANDARD_LINEWIDTH,
            color=FUNDAMENTAL_COLOR)
    ax.plot(rounds, last_trade_prices, label='Last Trade',
            color=PRICE_COLOR, alpha=0.8, linewidth=STANDARD_LINEWIDTH)
    ax.plot(rounds, midpoint_prices, label='Midpoint',
            color=MIDPOINT_COLOR, alpha=0.8, linewidth=STANDARD_LINEWIDTH)

    ax.set_xlabel('Round')
    ax.set_ylabel('Price')
    ax.set_title('Market Price Evolution')
    ax.legend(loc='best')
    ax.grid(True, alpha=GRID_ALPHA)

    return fig


def plot_bid_ask_spread(rounds: List[float], fundamental_prices: List[float],
                        midpoint_prices: List[float], best_bids: List[float],
                        best_asks: List[float]):
    """
    Plot bid-ask spread with fundamental value.

    Args:
        rounds: List of round numbers
        fundamental_prices: List of fundamental values
        midpoint_prices: List of midpoint prices
        best_bids: List of best bid prices
        best_asks: List of best ask prices

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    # Plot midpoint as main line
    ax.plot(rounds, midpoint_prices, label='Midpoint Price',
            color='blue', linewidth=STANDARD_LINEWIDTH)

    # Add fundamental value line
    ax.plot(rounds, fundamental_prices, label='Fundamental Value',
            linestyle=FUNDAMENTAL_LINESTYLE, linewidth=STANDARD_LINEWIDTH,
            color=FUNDAMENTAL_COLOR)

    # Plot bid and ask as filled area around midpoint
    ax.fill_between(rounds, best_bids, best_asks,
                    alpha=SPREAD_ALPHA, color=SPREAD_COLOR, label='Bid-Ask Spread')

    # Add actual bid and ask lines for clarity
    ax.plot(rounds, best_bids, '--', color=BID_COLOR, alpha=MEDIUM_ALPHA,
            label='Best Bid')
    ax.plot(rounds, best_asks, '--', color=ASK_COLOR, alpha=MEDIUM_ALPHA,
            label='Best Ask')

    ax.set_xlabel('Round')
    ax.set_ylabel('Price')
    ax.set_title('Price and Bid-Ask Spread')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)

    return fig


def plot_net_short_exposure(rounds: List[float], short_interest: List[float]):
    """
    Plot net short exposure over time.

    Args:
        rounds: List of round numbers
        short_interest: List of short interest values

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

    ax.plot(rounds, [-s for s in short_interest], label='Net Short Exposure',
            color=SHORT_EXPOSURE_COLOR, linewidth=STANDARD_LINEWIDTH)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=STANDARD_ALPHA)

    ax.set_xlabel('Round')
    ax.set_ylabel('Shares')
    ax.set_title('Net Short Exposure Over Time')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)

    return fig
