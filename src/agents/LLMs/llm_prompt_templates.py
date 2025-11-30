TRADING_OPTIONS_TEMPLATE = """
Your analysis should include:
- valuation_reasoning: Your numerical analysis of the asset's fundamental value
- valuation: Your estimate of the asset's current fundamental value
- price_prediction_reasoning: Your brief reasoning for your price predictions for the next 3 rounds
- price_prediction_t: Your predicted average transaction price for THIS round
- price_prediction_t1: Your predicted average transaction price for NEXT round
- price_prediction_t2: Your predicted average transaction price for the round AFTER next
- reasoning: Your explanation for the trading decision

Trading Options:
1. New Orders (replace_decision='Add'):
   - Single or multiple orders allowed
   - For each order:
     - Market order: Set order_type='market'
     - Limit order: Set order_type='limit' and specify price_limit
   - IMPORTANT: Sell orders may require borrowing shares if you don't hold enough
   {short_selling_note}

2. Cancel Orders (replace_decision='Cancel'):
   - Return an empty orders list: orders=[]

Your decision must include:
- orders: list of orders (empty list for Hold/Cancel)
  - For Buy/Sell orders, each must contain:
    - decision: "Buy" or "Sell"
    - quantity: number of shares
    - order_type: "market" or "limit"
    - price_limit: required for limit orders
- reasoning: brief explanation
- replace_decision: "Add", "Cancel", or "Replace"
"""

POSITION_INFO_TEMPLATE = """
Your Position:
- Available Shares: {shares} shares (negative indicates a short position)
- Main Cash Account: ${cash:.2f}
- Dividend Cash Account (not available for trading): ${dividend_cash:.2f}
- Total Available Cash: ${total_available_cash:.2f}{leverage_note}
- Shares in Orders: {committed_shares} shares
- Cash in Orders: ${committed_cash:.2f}
"""

LEVERAGE_INFO_TEMPLATE = """
Leverage Information:
- Leverage Enabled: {leverage_ratio:.1f}x maximum
- Borrowed Cash: ${borrowed_cash:.2f}
- Interest Paid on Leverage: ${leverage_interest_paid:.2f}
- Equity (Net Worth): ${equity:.2f}
- Gross Position Value: ${gross_position_value:.2f}
- Current Margin Ratio: {leverage_margin_ratio:.2%}
- Maintenance Margin Threshold: {maintenance_margin:.2%} (forced liquidation below this)
- Available Borrowing Power: ${available_borrowing_power:.2f}
- Status: {margin_status}
"""

DIVIDEND_INFO_TEMPLATE = """
Dividend Information:
Current Status:
- Last Paid Dividend: {last_paid_text}
- Expected Dividend: ${expected_dividend:.2f}

Dividend Model:
- Base Dividend: ${base_dividend:.2f}
- Variation Amount: ${variation:.2f}
- Maximum Scenario: ${max_dividend:.2f}  with {probability_percent:.0f}% probability
- Minimum Scenario: ${min_dividend:.2f} with {inverse_probability_percent:.0f}% probability

Payment Schedule:
- Next Payment in: {next_payment_round} rounds
- Payment Destination: {dividend_destination} account ({dividend_tradeable})
"""

# Template for REALIZATIONS_ONLY mode - only shows past payments
DIVIDEND_INFO_REALIZATIONS_TEMPLATE = """
Dividend Information:
- Last Paid Dividend: {last_paid_text}
- Number of Past Payments: {num_payments}
- Past Dividend Payments: {dividend_history_text}

Payment Schedule:
- Next Payment in: {next_payment_round} rounds
- Payment Destination: {dividend_destination} account ({dividend_tradeable})

Note: The dividend amount varies each period. You must estimate the distribution from past observations.
"""

# Template for AVERAGE mode - shows summary statistics
DIVIDEND_INFO_AVERAGE_TEMPLATE = """
Dividend Information:
- Last Paid Dividend: {last_paid_text}
- Average Dividend (from {num_payments} payments): {dividend_average_text}
- Standard Deviation: {dividend_std_text}

Payment Schedule:
- Next Payment in: {next_payment_round} rounds
- Payment Destination: {dividend_destination} account ({dividend_tradeable})

Note: Statistics are based on observed dividend payments.
"""

# Template for NONE mode - no dividend information
DIVIDEND_INFO_NONE_TEMPLATE = """
Dividend Information:
- Dividend information is not available for this market.
"""

INTEREST_INFO_TEMPLATE = """
Interest Rate Information:
- Base Rate: {interest_rate:.1f}%
- Compound Frequency: {compound_frequency} times per round
- Payment Destination: {interest_destination} account ({interest_tradeable})
"""

REDEMPTION_INFO_TEMPLATE = """
Redemption Information:
{redemption_text}
"""

# Standard user prompt template for all agents
STANDARD_USER_TEMPLATE = """{base_market_state}
{news_info}
{multi_stock_info}
{price_history}
{dividend_info}
{redemption_info}
{interest_info}
{trading_options}
{position_info}
{leverage_info}

Consider carefully the trade-offs between:
The execution uncertainty of limit orders and potential opportunity costs of holding an overvalued asset and missing on the interest rate or holding cash and missing on the dividend of an undervalued asset.

Your optimal decision should balance these factors based on your analysis.
Based on your trading strategy, what is your decision?"""


BASE_MARKET_TEMPLATE = """
Market State:
- Last Price: ${price:.2f}
- Round Number: {round_number}/{num_rounds}
- Best Public Estimate of Risk-Neutral Fundamental Value: {fundamental_display}
- Last Trading Volume: {volume_display}
- Price/Fundamental Ratio: {pf_ratio_display}

- Recent Trades
{trade_history}

Market Depth:
{order_book_display}
{orders_display}
"""

PRICE_HISTORY_TEMPLATE = """
Price History (last 5 rounds):
{price_history}
"""