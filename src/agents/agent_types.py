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