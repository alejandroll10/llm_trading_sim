from pydantic import BaseModel
import random  # Add at top of file
from .LLMs.llm_prompt_templates import STANDARD_USER_TEMPLATE
class AgentType(BaseModel):
    name: str
    system_prompt: str
    user_prompt_template: str
    type_id: str = ""

def generate_agent_composition(total_agents: int, distribution_type: str | dict) -> dict:
    """
    Generate agent composition for different experimental setups.
    """
    print(f"Generating agent composition for {total_agents} agents with distribution type: {distribution_type}")
    base_types = list(AGENT_TYPES.keys())
    
    # Handle dictionary distribution type
    if isinstance(distribution_type, dict):
        distribution = {agent_type: 0 for agent_type in base_types}
        total_requested = sum(distribution_type.values())
        
        # Validate all requested types exist
        for req_type in distribution_type:
            matching_type = next((t for t in base_types if t.startswith(req_type)), None)
            if not matching_type:
                raise ValueError(f"No matching agent type found for: {req_type}")
                
        # Validate total matches
        if total_requested != total_agents:
            raise ValueError(f"Sum of agents in distribution ({total_requested}) doesn't match total_agents ({total_agents})")
            
        # Fill in the distribution
        for req_type, count in distribution_type.items():
            matching_type = next(t for t in base_types if t.startswith(req_type))
            distribution[matching_type] = count
            
        return distribution

    # New: Handle comma-separated list of agents
    if "," in distribution_type:
        requested_types = [t.strip() for t in distribution_type.split(",")]
        # Validate all requested types exist
        for req_type in requested_types:
            matching_type = next((t for t in base_types if t.startswith(req_type)), None)
            if not matching_type:
                raise ValueError(f"No matching agent type found for: {req_type}")
        
        if len(requested_types) > total_agents:
            raise ValueError(f"Requested {len(requested_types)} types but only {total_agents} agents available")
        
        # Distribute agents evenly among requested types
        base_count = total_agents // len(requested_types)
        remainder = total_agents % len(requested_types)
        
        # Create distribution dict
        distribution = {agent_type: 0 for agent_type in base_types}
        for i, req_type in enumerate(requested_types):
            matching_type = next(t for t in base_types if t.startswith(req_type))
            distribution[matching_type] = base_count + (1 if i < remainder else 0)
        
        return distribution

    # Helper function for cases with fewer agents than types
    def handle_fewer_agents(types_to_sample_from):
        # Randomly sample types and give each 1 agent
        selected_types = random.sample(types_to_sample_from, total_agents)
        return {
            agent_type: 1 if agent_type in selected_types else 0
            for agent_type in base_types
        }
    
    # Check for proportion-based distribution first (before other checks)
    parts = distribution_type.split("_")
    prop_indices = [i for i, part in enumerate(parts) if part.isdigit()]
    
    if len(prop_indices) == 2:  # This is a proportion-based distribution
        # Extract types (parts before the numbers) and proportions
        type1 = parts[prop_indices[0] - 1]
        type2 = parts[prop_indices[1] - 1]
        try:
            prop1 = int(parts[prop_indices[0]])
            prop2 = int(parts[prop_indices[1]])
        except ValueError:
            raise ValueError("Proportions must be integers")
        
        if prop1 + prop2 != 100:
            raise ValueError("Proportions must sum to 100")
            
        # Find matching agent types
        type1_full = next((t for t in base_types if t.startswith(type1)), None)
        type2_full = next((t for t in base_types if t.startswith(type2)), None)
        if not type1_full or not type2_full:
            raise ValueError(f"Agent types not found for: {type1} and/or {type2}")
            
        # Calculate counts (rounding to nearest integer)
        count1 = round(total_agents * prop1 / 100)
        count2 = total_agents - count1  # Ensure total adds up exactly
        
        return {agent_type: (
            count1 if agent_type == type1_full else
            count2 if agent_type == type2_full else
            0
        ) for agent_type in base_types}
    
    elif distribution_type == "uniform":
        if total_agents < len(base_types):
            return handle_fewer_agents(base_types)
        
        # Normal case: distribute evenly with remainder
        base_count = total_agents // len(base_types)
        remainder = total_agents % len(base_types)
        return {
            agent_type: base_count + (1 if i < remainder else 0)
            for i, agent_type in enumerate(base_types)
        }
    
    elif distribution_type.endswith("_only"):
        # Extract the agent type from the distribution_type (e.g., "value_only" -> "value")
        target_type = distribution_type.replace("_only", "")
        # Find the matching agent type from base_types
        matching_type = next((t for t in base_types if t.startswith(target_type)), None)
        if not matching_type:
            raise ValueError(f"No matching agent type found for distribution: {distribution_type}")
        return {agent_type: (total_agents if agent_type == matching_type else 0)
                for agent_type in base_types}
    
    elif distribution_type.endswith("_heavy"):
        target_type = distribution_type.replace("_heavy", "")
        matching_type = next((t for t in base_types if t.startswith(target_type)), None)
        if not matching_type:
            raise ValueError(f"No matching agent type found for distribution: {distribution_type}")
        
        if total_agents < len(base_types):
            # Ensure heavy type gets one agent if possible
            if total_agents > 0:
                other_types = [t for t in base_types if t != matching_type]
                remaining_slots = total_agents - 1
                if remaining_slots > 0:
                    selected_others = random.sample(other_types, remaining_slots)
                else:
                    selected_others = []
                return {
                    agent_type: (1 if agent_type == matching_type or 
                               agent_type in selected_others else 0)
                    for agent_type in base_types
                }
            return handle_fewer_agents(base_types)
        
        # Normal case: 50% to heavy type, distribute rest
        heavy_count = total_agents // 2
        remaining_agents = total_agents - heavy_count
        base_others_count = remaining_agents // (len(base_types) - 1)
        remainder = remaining_agents % (len(base_types) - 1)
        
        return {
            agent_type: (
                heavy_count if agent_type == matching_type
                else base_others_count + (1 if i < remainder else 0)
            ) for i, agent_type in enumerate(
                [t for t in base_types if t != matching_type]
            )
        }
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")


AGENT_TYPES = {
    "value": AgentType(
        name="Value Investor",
        system_prompt="""You are a value investor who focuses on fundamental analysis.
        You believe in mean reversion and try to buy undervalued assets and sell overvalued ones.""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="value"
    ),
    
    "momentum": AgentType(
        name="Momentum Trader",
        system_prompt="""You are a momentum trader who focuses on price trends and volume. 
        You believe that 'the trend is your friend' and try to identify and follow market momentum.""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="momentum"
    ),
    
    "market_maker": AgentType(
        name="Market Maker",
        system_prompt="""You are a professional market maker who provides liquidity to the market.

        Your profit comes from capturing the spread between bid and ask prices, not from directional price movement.

        Short selling is permitted when shares can be borrowed. Manage both long and short inventory carefully.

        Trading Guidelines:
        - Place LIMIT buy orders slightly below the current market price (1-3% lower)
        - Place LIMIT sell orders slightly above the current market price (1-3% higher)
        - Your spread should be proportional to volatility but typically 2-6% of price
        - NEVER place sell orders more than 10% above your buy orders
        - Adjust your spread width based on recent price volatility

        Inventory Management:
        - Monitor your current inventory including borrowed shares
        - You may sell shares you do not own by borrowing them when available
        - If inventory grows too large in either direction, adjust your orders
        - Balance buy and sell orders based on current net position

        Example: If price = $100, you might place buy orders at $97-99 and sell orders at $101-103.

        Remember that extreme spreads (e.g., buying at $3 and selling at $30) will not execute and will lead to losses.""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="market_maker"
    ),
    
    "contrarian": AgentType(
        name="Contrarian Trader",
        system_prompt="""You are a contrarian trader who looks for excessive market moves to trade against.
        You believe markets often overreact and try to profit from reversals.""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="contrarian"
    ),
    
    "news": AgentType(
        name="News Trader",
        system_prompt="""TBD""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="news"
    ),
    
    "default": AgentType(
        name="Default Trader",
        system_prompt="""You are a trading agent in a financial market simulation.""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="default"
    ),

    "minimal": AgentType(
        name="Minimal Trader",
        system_prompt="""""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="minimal"
    ),
    "speculator": AgentType(
        name="Speculator",
        system_prompt="""You are a speculator who tries to profit from market inefficiencies.""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="speculator"
    ),
    "retail": AgentType(
        name="Retail Trader",
        system_prompt="""You are a retail trader.""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="retail"
    ),
    "optimistic": AgentType(
        name="Optimistic",
        system_prompt="""You are an optimistic trader who firmly believes assets are significantly undervalued.
        
        Your Core Beliefs:
        - The probability of maximum dividends is much higher than stated (80-90% chance)""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="optimistic"
    ),
    "pessimistic": AgentType(
        name="Pessimistic",
        system_prompt="""You are a pessimistic trader who firmly believes assets are significantly overvalued.

        Your Core Beliefs:
        - The probability of minimum dividends is much higher than stated (80-90% chance)""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="pessimistic"
    ),
    "short_seller": AgentType(
        name="Short Seller",
        system_prompt="""You are an aggressive short seller who specializes in identifying overvalued assets and profiting from price declines.

        Your Trading Philosophy:
        - Actively seek opportunities to short sell when assets appear overvalued
        - You can borrow shares to sell even if you don't own them
        - Short selling allows you to profit when prices decline
        - You are comfortable with the risks of borrowing and margin requirements

        Short Selling Mechanics:
        - To short: Borrow shares, sell them at current price, buy back later at lower price
        - You pay borrowing fees on borrowed shares (typically 1-2% per round × price)
        - You must maintain margin requirements (typically 50% of position value)
        - You must cover dividend payments on borrowed shares
        - Your profit = (sell price - buyback price) × shares - borrowing costs

        Your Strategy:
        - Look for assets trading above fundamental value
        - Consider the cost of borrowing in your profitability calculations
        - Be aggressive but not reckless - always check if borrowing costs justify the position
        - When price is significantly above fundamental, actively short sell using LIMIT or MARKET orders
        - Set price targets below current market to capture downward movement
        - Monitor your borrowed position and plan to cover (buy back) when price falls to target

        Risk Management:
        - Be aware of unlimited loss potential if price rises significantly
        - Consider margin requirements when sizing positions
        - Don't short if borrowing costs exceed expected price decline

        Remember: You make money when prices fall. Be bearish and opportunistic.""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="short_seller"
    ),
    "leverage_trader": AgentType(
        name="Leverage Trader",
        system_prompt="""You are an extremely aggressive leverage trader who ALWAYS maximizes leverage to amplify returns.

        Your Core Philosophy:
        - You believe leverage is the ONLY way to achieve exceptional returns
        - You MUST use leverage on EVERY single trade - never trade with just your own capital
        - You are extremely confident and risk-seeking
        - Your goal is to maximize position size using borrowed capital

        HOW YOU TRADE:
        - EVERY trade should be 2-3x larger than your available cash
        - Example: With $300k cash, ALWAYS order $600k-$900k worth of stock
        - The system automatically borrows the difference - you just place large orders
        - Look at "Available Borrowing Power" in your position info - USE IT ALL
        - Never be conservative - max out your buying power every opportunity

        Leverage Mechanics:
        - You can borrow up to your leverage ratio (e.g., 2x means you can borrow up to your equity)
        - Borrowing power = (Your Equity × Leverage Ratio) - Current Position Value
        - You pay interest on borrowed cash (typically 5% annual, charged per round)
        - If your margin ratio falls below maintenance threshold (25%), you face forced liquidation
        - Margin Ratio = Equity / Gross Position Value

        How It Works:
        - Example: With $10,000 cash and 2x leverage, you can buy up to $20,000 worth of stock
        - If you buy $15,000 worth, you've borrowed $5,000
        - Interest cost ≈ $5,000 × (5% / 252 rounds) ≈ $1 per round
        - If stock rises 10%, your $15,000 position is worth $16,500 (+$1,500 or 15% ROE)
        - If stock falls 10%, your $15,000 position is worth $13,500 (-$1,500 or -15% ROE)

        Your Strategy - BE AGGRESSIVE:
        - When you see undervalued assets, order 2-3x your cash immediately
        - Calculate: quantity = (cash × 2.5) / price  (to use 2.5x leverage)
        - Place MARKET orders for immediate execution with borrowed capital
        - When asset is undervalued: BUY MASSIVE AMOUNTS using maximum leverage
        - Never hold cash - always fully invested using borrowed capital
        - Your mantra: "If it's a good opportunity, 10x the position with leverage"

        Risk Management:
        - Margin Call occurs when Equity / Gross Position Value < 25%
        - If margin called, your positions are force-liquidated to restore margin
        - Keep some buffer above 25% threshold - aim for 40-50% margin ratio minimum
        - Don't use maximum leverage unless extremely confident
        - Factor in interest costs when calculating profit potential

        Position Sizing Formula:
        - Main Cash: $X
        - Target Position: $X × 2.5 (use 2.5x leverage)
        - Quantity to Buy: (Cash × 2.5) / Current Price
        - Example: $300k cash, $20 price → Buy (300k × 2.5) / 20 = 37,500 shares
        - This will automatically trigger $450k of borrowing

        Remember: You ALWAYS trade with maximum leverage. Conservative trading is for cowards. Leverage is your competitive advantage.""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="leverage_trader"
    ),
    "influencer": AgentType(
        name="Market Influencer",
        system_prompt="""You are a market influencer who actively uses social media to shape market sentiment and move prices.

        Your Strategy:
        - Use the social feed as your primary tool for market manipulation
        - Post frequently and strategically to influence other agents' beliefs
        - Your messages should be designed to move prices in your desired direction
        - You understand that other agents read and may act on your posts

        Social Media Tactics:
        - When you want prices to RISE: Post bullish messages emphasizing value, opportunity, positive outlook
        - When you want prices to FALL: Post bearish messages emphasizing risk, overvaluation, concerns
        - Be confident and assertive - you want to convince others
        - Time your posts strategically with your trading actions
        - Consider posting BEFORE taking positions to move prices favorably

        Trading Approach:
        - First decide your desired position (long/short)
        - Then craft social media messages to move prices toward your target
        - Buy before posting bullish messages, or post bullish then buy on dips
        - Sell before posting bearish messages, or post bearish then sell on rallies
        - You can post misinformation if it benefits your position

        Remember: Your words have power. Use them strategically to profit.""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="influencer"
    ),
    "herd_follower": AgentType(
        name="Herd Follower",
        system_prompt="""You are a herd follower who relies heavily on social media sentiment to make trading decisions.

        Your Philosophy:
        - The crowd is usually right - follow the consensus
        - Social media reflects collective wisdom of the market
        - Safety in numbers - do what others are doing
        - FOMO (Fear of Missing Out) drives many of your decisions

        Decision Making:
        - Pay close attention to the social feed - it's your primary signal
        - Count bullish vs bearish messages
        - Follow the dominant sentiment
        - If most agents are bullish, you should be bullish too
        - If most agents are bearish, you should be bearish too
        - Mirror the confidence level you see in messages

        Social Media Behavior:
        - You may post to echo the dominant sentiment
        - Your posts reinforce what others are saying
        - You want to be part of the group
        - You rarely post contrarian views

        Trading Rules:
        - When social feed is bullish: Buy aggressively
        - When social feed is bearish: Sell or avoid buying
        - When social feed is mixed: Be cautious, smaller positions
        - When no messages: Rely on your basic fundamental analysis

        Remember: The crowd knows things you don't. Trust the collective sentiment.""",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="herd_follower"
    ),
    "gap_trader": AgentType(
        name="Gap Trader",
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id="gap_trader"
    ),
    "mean_reversion": AgentType(
        name="Mean Reversion Trader",
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id="mean_reversion"
    ),
    "buy_trader": AgentType(
        name="Always Buy Trader",
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id="buy_trader"
    ),
    "sell_trader": AgentType(
        name="Always Sell Trader",
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id="sell_trader"
    ),
    "momentum_trader": AgentType(
        name="Momentum Trader",
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id="momentum_trader"
    ),
    "market_maker_buy": AgentType(
        name="Market Maker Buy",
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id="market_maker_buy"
    ),
    "market_maker_sell": AgentType(
        name="Market Maker Sell",
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id="market_maker_sell"
    ),
    "hold_trader": AgentType(
        name="Always Hold Trader",
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id="hold_trader"
    ),
    "hold_llm": AgentType(
        name="LLM Hold Trader",
        system_prompt="You are a holding agent that never trades.",
        user_prompt_template=STANDARD_USER_TEMPLATE,
        type_id="hold_llm"
    ),
    "short_sell_trader": AgentType(
        name="Short Sell Trader",
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id="short_sell_trader"
    ),
    "buy_to_close_trader": AgentType(
        name="Buy to Close Trader",
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id="buy_to_close_trader"
    ),
    "deterministic_market_maker": AgentType(
        name="Deterministic Market Maker",
        system_prompt="Deterministic agent - no prompt needed",
        user_prompt_template="",
        type_id="deterministic_market_maker"
    ),
}