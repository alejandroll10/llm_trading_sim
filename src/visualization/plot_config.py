"""Configuration for plot styling and parameters."""

# Figure sizes
STANDARD_FIGSIZE = (12, 6)
LARGE_FIGSIZE = (12, 8)
EXTRA_LARGE_FIGSIZE = (14, 8)
TALL_FIGSIZE = (12, 10)
WORDCLOUD_FIGSIZE = (12, 8)

# Colors
WEALTH_COLORS = ['#3498db', '#2ecc71', '#e74c3c']  # Cash, Dividend, Shares
FUNDAMENTAL_COLOR = 'green'
PRICE_COLOR = 'red'
MIDPOINT_COLOR = 'purple'
BID_COLOR = 'red'
ASK_COLOR = 'green'
SPREAD_COLOR = 'gray'
SHORT_EXPOSURE_COLOR = 'blue'

# Transparency
STANDARD_ALPHA = 0.7
LIGHT_ALPHA = 0.3
MEDIUM_ALPHA = 0.6
SPREAD_ALPHA = 0.2

# Line styles
FUNDAMENTAL_LINESTYLE = '--'
STANDARD_LINEWIDTH = 2
THIN_LINEWIDTH = 1.5

# Grid
GRID_ALPHA = 0.3

# WordCloud parameters
WORDCLOUD_WIDTH = 1200
WORDCLOUD_HEIGHT = 800
WORDCLOUD_BACKGROUND = 'white'
WORDCLOUD_MIN_FONT = 10

# Heatmap parameters
HEATMAP_COLORMAP = 'RdYlGn'
HEATMAP_VMIN = -1
HEATMAP_VMAX = 1
HEATMAP_INTERPOLATION = 'nearest'

# Risk-free rate for excess return calculations
PER_ROUND_RISK_FREE_RATE = 0.001  # 0.1% per round
