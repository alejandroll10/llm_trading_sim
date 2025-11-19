from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal
import openai
from agents.agents_api import TradeDecision, OrderDetails
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from scenarios.base import DEFAULT_LLM_BASE_URL

class OrderSchema(BaseModel):
    """Schema for individual orders"""
    decision: Literal["Buy", "Sell"] = Field(..., description="Buy or Sell")
    quantity: int = Field(..., description="Number of shares")
    order_type: str = Field(..., description="Market or Limit")
    price_limit: Optional[float] = Field(None, description="Required for limit orders")
    stock_id: str = Field(default="DEFAULT_STOCK", description="Stock identifier (automatically set for single-stock scenarios)")

class TradeDecisionSchema(BaseModel):

    """Schema for trade decisions"""
    valuation_reasoning: str = Field(..., description="Brief numerical calculation of valuation analysis")
    valuation: float = Field(..., description="Agent's estimated fundamental value")
    price_target_reasoning: str = Field(..., description="Specific reasoning of expected price next round")
    price_target: float = Field(..., description="Agent's predicted price in near future")
    reasoning: str = Field(..., description="Your strategy and reasoning for this trade - decide BEFORE specifying orders")
    orders: List[OrderSchema] = Field(..., description="List of orders to execute")
    replace_decision: str = Field(..., description="Add, Cancel, or Replace")
    message_reasoning: Optional[str] = Field(None, description="Your reasoning for this social media message - what effect do you want it to have on other agents?")
    post_message: Optional[str] = Field(None, description="Optional: Post a message to the social feed visible to other agents next round")
    notes_to_self: Optional[str] = Field(None, description="Optional: Write notes to your future self about what you learned this round, patterns observed, or strategy adjustments. These notes will be shown to you in future rounds.")
    

@dataclass
class LLMRequest:
    """Structure for LLM request"""
    system_prompt: str
    user_prompt: str
    model: str
    agent_id: str
    round_number: int
    is_multi_stock: bool = False

@dataclass
class LLMResponse:
    """Structure for LLM response"""
    decision: Dict[str, Any]
    raw_response: str

class LLMService:
    """Pure service for LLM interactions"""
    
    def __init__(self):
        load_dotenv()  # Load API key from .env

        # Initialize OpenAI client with configured base_url
        # (base_url set in scenarios/base.py - non-sensitive config)
        if DEFAULT_LLM_BASE_URL:
            self.client = openai.OpenAI(base_url=DEFAULT_LLM_BASE_URL)
        else:
            self.client = openai.OpenAI()  # Use default OpenAI endpoint

        self.seed = 42
    
    def get_decision(self, request: LLMRequest) -> LLMResponse:
        """Get decision from LLM"""
        # Special case for hold_llm agent type to avoid API calls
        if request.model == "hold_llm":
            decision = TradeDecision(
                valuation_reasoning="LLM Hold Agent: Always hold strategy",
                valuation=0,
                price_target=0,
                price_target_reasoning="LLM Hold Agent: Always hold strategy",
                orders=[],  # Empty list for hold
                replace_decision="Add",
                reasoning="LLM Hold Agent: Always hold strategy"
            ).model_dump()
            decision["agent_id"] = request.agent_id
            
            return LLMResponse(
                decision=decision,
                raw_response="Testing hold_llm agent. Always holds."
            )
            
        # Normal LLM processing for other models
        # Conditionally append multi-stock instructions
        user_prompt = request.user_prompt
        if request.is_multi_stock:
            multi_stock_instructions = """
IMPORTANT: This is a MULTI-STOCK scenario. You MUST include stock_id for each order using the stock IDs shown in the market information (e.g., "TECH_A", "TECH_B")."""
            user_prompt = user_prompt + multi_stock_instructions

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Get the response using the parse method with our schema
        completion = self.client.beta.chat.completions.parse(
            model=request.model,
            messages=messages,
            response_format=TradeDecisionSchema,
            temperature=0.0,
            seed=self.seed
        )
        
        # Get raw response from LLM
        raw_response = completion.choices[0].message.content
        
        # Parse the response into structured format
        try:
            parsed_response = completion.choices[0].message.parsed
            
            # Convert the parsed response to OrderDetails format
            orders = [OrderDetails(**order.model_dump()) for order in parsed_response.orders]
            
            # Create TradeDecision
            decision = TradeDecision(
                valuation_reasoning=parsed_response.valuation_reasoning,
                valuation=parsed_response.valuation,
                price_target_reasoning=parsed_response.price_target_reasoning,
                price_target=parsed_response.price_target,
                orders=orders,
                replace_decision=parsed_response.replace_decision,
                reasoning=parsed_response.reasoning,
                notes_to_self=parsed_response.notes_to_self  # Optional field
            ).model_dump()
            decision["agent_id"] = request.agent_id
            # Add optional message_reasoning if provided
            if parsed_response.message_reasoning:
                decision["message_reasoning"] = parsed_response.message_reasoning
            # Add optional post_message if provided
            if parsed_response.post_message:
                decision["post_message"] = parsed_response.post_message

            return LLMResponse(
                raw_response=raw_response,
                decision=decision
            )
        except Exception as e:
            # Still return the raw response even if parsing fails
            return LLMResponse(
                raw_response=raw_response,
                decision=self.get_fallback_decision(request.agent_id)
            )
    
    def get_fallback_decision(self, agent_id: str) -> Dict[str, Any]:
        """Get fallback decision when LLM fails"""
        fallback = TradeDecision(
            valuation_reasoning="Fallback decision due to parsing error",
            valuation=0,
            price_target=0,
            price_target_reasoning="Fallback decision due to parsing error",
            orders=[],  # Empty list for hold
            replace_decision="Add",
            reasoning="Fallback decision due to parsing error",
            notes_to_self=None
        ).model_dump()
        fallback["agent_id"] = agent_id
        return fallback
