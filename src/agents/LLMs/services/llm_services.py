from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal
import openai
from agents.agents_api import TradeDecision, OrderDetails
from pydantic import BaseModel, Field
from dotenv import load_dotenv

class OrderSchema(BaseModel):
    """Schema for individual orders"""
    decision: Literal["Buy", "Sell"] = Field(..., description="Buy or Sell")
    quantity: int = Field(..., description="Number of shares")
    order_type: str = Field(..., description="Market or Limit")
    price_limit: Optional[float] = Field(None, description="Required for limit orders")

class TradeDecisionSchema(BaseModel):
    
    """Schema for trade decisions"""
    valuation_reasoning: str = Field(..., description="Brief numerical calculation of valuation analysis")
    valuation: float = Field(..., description="Agent's estimated fundamental value")
    price_target_reasoning: str = Field(..., description="Specific reasoning of expected price next round")
    price_target: float = Field(..., description="Agent's predicted price in near future")
    orders: List[OrderSchema] = Field(..., description="List of orders to execute")
    replace_decision: str = Field(..., description="Add, Cancel, or Replace")
    reasoning: str = Field(..., description="Explanation for the trading decisions")
    

@dataclass
class LLMRequest:
    """Structure for LLM request"""
    system_prompt: str
    user_prompt: str
    model: str
    agent_id: str
    round_number: int

@dataclass
class LLMResponse:
    """Structure for LLM response"""
    decision: Dict[str, Any]
    raw_response: str

class LLMService:
    """Pure service for LLM interactions"""
    
    def __init__(self):
        load_dotenv()
        self.client = openai.OpenAI()
        self.seed = 42
    
    def get_decision(self, request: LLMRequest) -> LLMResponse:
        """Get decision from LLM"""
        # Special case for hold_llm agent type to avoid API calls
        if request.model == "hold_llm":
            decision = TradeDecision(
                valuation_reasoning="LLM Hold Agent: Always hold strategy",
                valuation=0,
                price_target=0,
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
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt}
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
                reasoning=parsed_response.reasoning
            ).model_dump()
            decision["agent_id"] = request.agent_id
            
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
            reasoning="Fallback decision due to parsing error"
        ).model_dump()
        fallback["agent_id"] = agent_id
        return fallback