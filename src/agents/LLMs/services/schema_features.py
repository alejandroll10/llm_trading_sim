"""
Feature Toggle System for Agent Capabilities

This module implements a composable feature registry that enables/disables
agent capabilities (memory, social media, etc.) without combinatorial explosion
of Pydantic schemas.

Architecture:
    Feature Configuration → Feature Registry → Dynamic Schema Generation
            ↓                      ↓                       ↓
        Scenarios          Field Definitions        Pydantic Model

Benefits:
    - Scalable: N features, not 2^N classes
    - Composable: Mix and match any features
    - Type-safe: Pydantic validates correctly
    - Maintainable: One place to add features
    - A/B testable: Easy to toggle features on/off
"""

from enum import Enum
from typing import Dict, Any, Type, Optional, List, Literal, Set
from pydantic import BaseModel, Field
from functools import lru_cache


class Feature(str, Enum):
    """Available agent features that can be toggled on/off"""
    MEMORY = "memory"
    SOCIAL = "social"
    LAST_REASONING = "last_reasoning"  # Show agent their reasoning from last round
    SELF_MODIFY = "self_modify"  # Allow agents to modify their own system prompts
    # Future features can be added here:
    # ADVANCED_ORDERS = "advanced_orders"
    # PORTFOLIO_ANALYSIS = "portfolio_analysis"
    # NEWS_FEED = "news_feed"


class FeatureRegistry:
    """
    Registry of field definitions for dynamic schema generation.

    This class maintains the mapping between features and their corresponding
    Pydantic field definitions, enabling dynamic schema composition.
    """

    @staticmethod
    def create_order_schema(is_multi_stock: bool = False) -> Type[BaseModel]:
        """
        Create OrderSchema dynamically based on whether it's a multi-stock scenario.

        In single-stock mode: stock_id field is EXCLUDED from schema
        In multi-stock mode: stock_id field is REQUIRED

        Args:
            is_multi_stock: Whether this is a multi-stock scenario

        Returns:
            Dynamically created OrderSchema class
        """
        base_fields = {
            'decision': (Literal["Buy", "Sell"], Field(..., description="Buy or Sell")),
            'quantity': (int, Field(..., description="Number of shares")),
            'order_type': (str, Field(..., description="Market or Limit")),
            'price_limit': (Optional[float], Field(None, description="Required for limit orders")),
        }

        # Only add stock_id field in multi-stock mode
        if is_multi_stock:
            base_fields['stock_id'] = (str, Field(..., description="Stock identifier - must match one of the stocks in the market"))

        # Create the dynamic OrderSchema model
        OrderSchema = type(
            "OrderSchema",
            (BaseModel,),
            {
                '__annotations__': {name: field_type for name, (field_type, _) in base_fields.items()},
                **{name: field_default for name, (_, field_default) in base_fields.items()}
            }
        )

        return OrderSchema

    @staticmethod
    def get_core_fields(OrderSchema: Type[BaseModel]) -> Dict[str, tuple]:
        """
        Get core fields with the appropriate OrderSchema.

        Args:
            OrderSchema: The dynamically created OrderSchema class

        Returns:
            Dict of core field definitions
        """
        return {
            'valuation_reasoning': (str, Field(..., description="Brief numerical calculation of valuation analysis")),
            'valuation': (float, Field(..., description="Agent's estimated fundamental value")),
            'price_target_reasoning': (str, Field(..., description="Specific reasoning of expected price next round")),
            'price_target': (float, Field(..., description="Agent's predicted price in near future")),
            'reasoning': (str, Field(..., description="Your strategy and reasoning for this trade - decide BEFORE specifying orders")),
            'orders': (List[OrderSchema], Field(..., description="List of orders to execute")),
            'replace_decision': (str, Field(..., description="Add, Cancel, or Replace")),
        }

    # Feature-specific fields that are conditionally added
    FEATURE_FIELDS: Dict[Feature, Dict[str, tuple]] = {
        Feature.MEMORY: {
            'notes_to_self': (
                Optional[str],
                Field(None, description="Optional: Write notes to your future self about what you learned this round, patterns observed, or strategy adjustments. These notes will be shown to you in future rounds.")
            ),
        },
        Feature.SOCIAL: {
            'message_reasoning': (
                Optional[str],
                Field(None, description="Your reasoning for this social media message - what effect do you want it to have on other agents?")
            ),
            'post_message': (
                Optional[str],
                Field(None, description="Optional: Post a message to the social feed visible to other agents next round")
            ),
        },
        Feature.SELF_MODIFY: {
            'prompt_modification': (
                Optional[str],
                Field(None, description="Optional: Propose a modification to your trading strategy/system prompt. This will update how you approach future trading decisions. Use this to evolve your strategy based on what you've learned.")
            ),
            'modification_reasoning': (
                Optional[str],
                Field(None, description="Why you want to modify your strategy - what have you learned that warrants this change?")
            ),
        },
    }

    @staticmethod
    def _create_feature_set_key(features: Set[Feature]) -> str:
        """Create a hashable key for caching based on feature set"""
        return ",".join(sorted(f.value for f in features))

    @staticmethod
    @lru_cache(maxsize=32)  # Cache schemas for all possible feature+stock combinations
    def create_trade_decision_model(features_key: str, is_multi_stock: bool = False) -> Type[BaseModel]:
        """
        Dynamically create a Pydantic model based on enabled features and stock mode.

        This method uses Pydantic's create_model() to generate a schema at runtime
        that includes only the fields for enabled features. The result is cached
        to avoid regenerating schemas for the same feature combination.

        Args:
            features_key: Comma-separated string of feature names (for caching)
            is_multi_stock: Whether this is a multi-stock scenario

        Returns:
            A Pydantic BaseModel class with the appropriate fields

        Example:
            >>> features = {Feature.MEMORY, Feature.SOCIAL}
            >>> key = FeatureRegistry._create_feature_set_key(features)
            >>> schema = FeatureRegistry.create_trade_decision_model(key, is_multi_stock=False)
            >>> # schema now has core fields + memory fields + social fields
        """
        # Parse features from key
        features = set()
        if features_key:
            features = {Feature(name) for name in features_key.split(",")}

        # Create appropriate OrderSchema for this stock mode
        OrderSchema = FeatureRegistry.create_order_schema(is_multi_stock)

        # Start with core fields (using the correct OrderSchema)
        field_definitions = FeatureRegistry.get_core_fields(OrderSchema)

        # Add feature-specific fields
        for feature in features:
            if feature in FeatureRegistry.FEATURE_FIELDS:
                field_definitions.update(FeatureRegistry.FEATURE_FIELDS[feature])

        # Create the dynamic model using Pydantic's create_model
        # The model name reflects which features are enabled for debugging
        feature_names = "_".join(sorted(f.value for f in features)) if features else "base"
        stock_mode = "multi" if is_multi_stock else "single"
        model_name = f"TradeDecisionSchema_{feature_names}_{stock_mode}"

        # Create model with all field definitions
        DynamicModel = type(
            model_name,
            (BaseModel,),
            {
                '__annotations__': {
                    name: field_type
                    for name, (field_type, _) in field_definitions.items()
                },
                **{
                    name: field_default
                    for name, (_, field_default) in field_definitions.items()
                }
            }
        )

        return DynamicModel

    @staticmethod
    def get_schema_for_features(features: Set[Feature], is_multi_stock: bool = False) -> Type[BaseModel]:
        """
        Convenience method to get a schema for a set of features and stock mode.

        Args:
            features: Set of enabled Feature enums
            is_multi_stock: Whether this is a multi-stock scenario

        Returns:
            A Pydantic BaseModel class with the appropriate fields
        """
        key = FeatureRegistry._create_feature_set_key(features)
        return FeatureRegistry.create_trade_decision_model(key, is_multi_stock)

    @staticmethod
    def get_all_features() -> Set[Feature]:
        """Get all available features (for backward compatibility mode)"""
        return {Feature.MEMORY, Feature.SOCIAL, Feature.LAST_REASONING}

    @staticmethod
    def extract_features_from_config(config: Dict[str, Any]) -> Set[Feature]:
        """
        Extract enabled features from scenario/agent configuration.

        Args:
            config: Configuration dict with feature flags (e.g., MEMORY_ENABLED, SOCIAL_ENABLED)

        Returns:
            Set of enabled Feature enums

        Example:
            >>> config = {"MEMORY_ENABLED": True, "SOCIAL_ENABLED": False}
            >>> features = FeatureRegistry.extract_features_from_config(config)
            >>> # features == {Feature.MEMORY}
        """
        features = set()

        # Check each feature flag
        if config.get('MEMORY_ENABLED', True):  # Default True for backward compatibility
            features.add(Feature.MEMORY)

        if config.get('SOCIAL_ENABLED', True):  # Default True for backward compatibility
            features.add(Feature.SOCIAL)

        if config.get('LAST_REASONING_ENABLED', True):  # Default True for backward compatibility
            features.add(Feature.LAST_REASONING)

        # Self-modify is opt-in (default False) - experimental feature
        if config.get('SELF_MODIFY_ENABLED', False):
            features.add(Feature.SELF_MODIFY)

        return features


# Backward compatibility: Export a default schema with all features enabled (single-stock)
# This allows existing code to continue working without changes
TradeDecisionSchema = FeatureRegistry.get_schema_for_features(
    FeatureRegistry.get_all_features(),
    is_multi_stock=False
)
