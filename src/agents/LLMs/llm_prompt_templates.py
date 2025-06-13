TRADING_OPTIONS_TEMPLATE = """
Your analysis should include:
- valuation_reasoning: Your numerical analysis of the asset's fundamental value
- valuation: Your estimate of the asset's current fundamental value
- price_target_reasoning: Your numerical analysis of the asset's price target
- price_target: Your predicted price for the next round
- reasoning: Your explanation for the trading decision

Trading Options:
1. New Orders (replace_decision='Add'):
   - Single or multiple orders allowed
   - For each order:
     - Market order: Set order_type='market'
     - Limit order: Set order_type='limit' and specify price_limit
   - IMPORTANT: Sell orders require sufficient available shares
   - Short selling is NOT allowed

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
- Available Shares: {shares} shares (Short selling is not allowed)
- Main Cash Account: ${cash:.2f}
- Dividend Cash Account (not available for trading): ${dividend_cash:.2f}
- Total Available Cash: ${total_available_cash:.2f} (Borrowing is not allowed)
- Shares in Orders: {committed_shares} shares
- Cash in Orders: ${committed_cash:.2f}
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
{price_history}
{dividend_info}
{redemption_info}
{interest_info}
{trading_options}
{position_info}

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