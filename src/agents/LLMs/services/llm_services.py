from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal, Set
import openai
import time
import logging
from agents.agents_api import TradeDecision, OrderDetails
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from scenarios.base import DEFAULT_LLM_BASE_URL
from .schema_features import Feature, FeatureRegistry

logger = logging.getLogger("llm_timing")

# NOTE: OrderSchema and TradeDecisionSchema are now dynamically generated
# in schema_features.py based on enabled features and stock mode.
# This allows for:
# - Conditional inclusion of optional fields (memory, social)
# - Conditional inclusion of stock_id (only in multi-stock mode)

@dataclass
class LLMRequest:
    """Structure for LLM request"""
    system_prompt: str
    user_prompt: str
    model: str
    agent_id: str
    round_number: int
    is_multi_stock: bool = False
    enabled_features: Set[Feature] = None  # NEW: Feature configuration for dynamic schema

    def __post_init__(self):
        """Set default features if none provided (backward compatibility)"""
        if self.enabled_features is None:
            self.enabled_features = FeatureRegistry.get_all_features()

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
        """Get decision from LLM using dynamic schema based on enabled features"""
        # Special case for hold_llm agent type to avoid API calls
        if request.model == "hold_llm":
            # Create hold decision with only the fields enabled by features
            decision_dict = {
                "valuation_reasoning": "LLM Hold Agent: Always hold strategy",
                "valuation": 0,
                "price_prediction_reasoning": "LLM Hold Agent: Always hold strategy",
                "price_prediction_t": 0,
                "price_prediction_t1": 0,
                "price_prediction_t2": 0,
                "orders": [],  # Empty list for hold
                "replace_decision": "Add",
                "reasoning": "LLM Hold Agent: Always hold strategy",
            }

            # Add optional fields only if features are enabled
            if Feature.MEMORY in request.enabled_features:
                decision_dict["notes_to_self"] = None
            if Feature.SOCIAL in request.enabled_features:
                decision_dict["message_reasoning"] = None
                decision_dict["post_message"] = None
            if Feature.SELF_MODIFY in request.enabled_features:
                decision_dict["prompt_modification"] = None
                decision_dict["modification_reasoning"] = None

            decision_dict["agent_id"] = request.agent_id

            return LLMResponse(
                decision=decision_dict,
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

        # Get dynamic schema based on enabled features AND stock mode
        dynamic_schema = FeatureRegistry.get_schema_for_features(
            request.enabled_features,
            is_multi_stock=request.is_multi_stock
        )

        # Get the response using the parse method with our dynamic schema
        # Use timeout + retry for flaky APIs (e.g., UF Hypergator)
        start_time = time.time()
        prompt_len = len(request.system_prompt) + len(request.user_prompt)

        max_retries = 3
        timeout_seconds = 60  # 60s timeout per attempt

        for attempt in range(max_retries):
            try:
                logger.warning(f"[LLM_CALL] Agent {request.agent_id} R{request.round_number}: Calling {request.model} (~{prompt_len//4} tokens){'...' if attempt == 0 else f' (retry {attempt})...'}")
                completion = self.client.beta.chat.completions.parse(
                    model=request.model,
                    messages=messages,
                    response_format=dynamic_schema,
                    temperature=0.0,
                    seed=self.seed,
                    timeout=timeout_seconds
                )
                elapsed = time.time() - start_time
                logger.warning(f"[LLM_CALL] Agent {request.agent_id} R{request.round_number}: Response in {elapsed:.1f}s")
                break  # Success, exit retry loop
            except Exception as e:
                elapsed = time.time() - start_time
                if attempt < max_retries - 1:
                    logger.warning(f"[LLM_CALL] Agent {request.agent_id} R{request.round_number}: Timeout/error after {elapsed:.1f}s, retrying...")
                    continue
                else:
                    logger.warning(f"[LLM_CALL] Agent {request.agent_id} R{request.round_number}: Failed after {max_retries} attempts ({elapsed:.1f}s total)")
                    raise e

        # Get raw response from LLM
        raw_response = completion.choices[0].message.content

        # Parse the response into structured format
        try:
            parsed_response = completion.choices[0].message.parsed

            # Convert the parsed response to OrderDetails format
            # In single-stock mode, add stock_id automatically since it's not in the schema
            orders = []
            for order in parsed_response.orders:
                order_dict = order.model_dump()
                # Add stock_id if not present (single-stock mode)
                if 'stock_id' not in order_dict:
                    order_dict['stock_id'] = 'DEFAULT_STOCK'
                orders.append(OrderDetails(**order_dict))

            # Build decision dict with core fields (always present)
            decision_dict = {
                "valuation_reasoning": parsed_response.valuation_reasoning,
                "valuation": parsed_response.valuation,
                "price_prediction_reasoning": parsed_response.price_prediction_reasoning,
                "price_prediction_t": parsed_response.price_prediction_t,
                "price_prediction_t1": parsed_response.price_prediction_t1,
                "price_prediction_t2": parsed_response.price_prediction_t2,
                "orders": orders,
                "replace_decision": parsed_response.replace_decision,
                "reasoning": parsed_response.reasoning,
            }

            # Add feature-specific fields only if enabled
            if Feature.MEMORY in request.enabled_features:
                decision_dict["notes_to_self"] = getattr(parsed_response, 'notes_to_self', None)

            if Feature.SOCIAL in request.enabled_features:
                decision_dict["message_reasoning"] = getattr(parsed_response, 'message_reasoning', None)
                decision_dict["post_message"] = getattr(parsed_response, 'post_message', None)

            if Feature.SELF_MODIFY in request.enabled_features:
                decision_dict["prompt_modification"] = getattr(parsed_response, 'prompt_modification', None)
                decision_dict["modification_reasoning"] = getattr(parsed_response, 'modification_reasoning', None)

            decision_dict["agent_id"] = request.agent_id

            return LLMResponse(
                raw_response=raw_response,
                decision=decision_dict
            )
        except Exception as e:
            # Still return the raw response even if parsing fails
            return LLMResponse(
                raw_response=raw_response,
                decision=self.get_fallback_decision(request.agent_id, request.enabled_features)
            )
    
    def get_fallback_decision(self, agent_id: str, enabled_features: Set[Feature] = None) -> Dict[str, Any]:
        """Get fallback decision when LLM fails, respecting enabled features"""
        if enabled_features is None:
            enabled_features = FeatureRegistry.get_all_features()

        # Build fallback with core fields
        fallback = {
            "valuation_reasoning": "Fallback decision due to parsing error",
            "valuation": 0,
            "price_prediction_reasoning": "Fallback decision due to parsing error",
            "price_prediction_t": 0,
            "price_prediction_t1": 0,
            "price_prediction_t2": 0,
            "orders": [],  # Empty list for hold
            "replace_decision": "Add",
            "reasoning": "Fallback decision due to parsing error",
        }

        # Add optional fields only if features are enabled
        if Feature.MEMORY in enabled_features:
            fallback["notes_to_self"] = None

        if Feature.SOCIAL in enabled_features:
            fallback["message_reasoning"] = None
            fallback["post_message"] = None

        if Feature.SELF_MODIFY in enabled_features:
            fallback["prompt_modification"] = None
            fallback["modification_reasoning"] = None

        fallback["agent_id"] = agent_id
        return fallback
