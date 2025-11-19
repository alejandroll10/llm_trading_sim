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
    # Future features can be added here:
    # ADVANCED_ORDERS = "advanced_orders"
    # PORTFOLIO_ANALYSIS = "portfolio_analysis"
    # NEWS_FEED = "news_feed"


class OrderSchema(BaseModel):
    """Schema for individual orders (always present)"""
    decision: Literal["Buy", "Sell"] = Field(..., description="Buy or Sell")
    quantity: int = Field(..., description="Number of shares")
    order_type: str = Field(..., description="Market or Limit")
    price_limit: Optional[float] = Field(None, description="Required for limit orders")
    stock_id: str = Field(default="DEFAULT_STOCK", description="Stock identifier (automatically set for single-stock scenarios)")


class FeatureRegistry:
    """
    Registry of field definitions for dynamic schema generation.

    This class maintains the mapping between features and their corresponding
    Pydantic field definitions, enabling dynamic schema composition.
    """

    # Core fields that are ALWAYS present in every schema
    CORE_FIELDS: Dict[str, tuple] = {
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
    }

    @staticmethod
    def _create_feature_set_key(features: Set[Feature]) -> str:
        """Create a hashable key for caching based on feature set"""
        return ",".join(sorted(f.value for f in features))

    @staticmethod
    @lru_cache(maxsize=16)  # Cache schemas for all possible feature combinations
    def create_trade_decision_model(features_key: str) -> Type[BaseModel]:
        """
        Dynamically create a Pydantic model based on enabled features.

        This method uses Pydantic's create_model() to generate a schema at runtime
        that includes only the fields for enabled features. The result is cached
        to avoid regenerating schemas for the same feature combination.

        Args:
            features_key: Comma-separated string of feature names (for caching)

        Returns:
            A Pydantic BaseModel class with the appropriate fields

        Example:
            >>> features = {Feature.MEMORY, Feature.SOCIAL}
            >>> key = FeatureRegistry._create_feature_set_key(features)
            >>> schema = FeatureRegistry.create_trade_decision_model(key)
            >>> # schema now has core fields + memory fields + social fields
        """
        # Parse features from key
        features = set()
        if features_key:
            features = {Feature(name) for name in features_key.split(",")}

        # Start with core fields
        field_definitions = dict(FeatureRegistry.CORE_FIELDS)

        # Add feature-specific fields
        for feature in features:
            if feature in FeatureRegistry.FEATURE_FIELDS:
                field_definitions.update(FeatureRegistry.FEATURE_FIELDS[feature])

        # Create the dynamic model using Pydantic's create_model
        # The model name reflects which features are enabled for debugging
        feature_names = "_".join(sorted(f.value for f in features)) if features else "base"
        model_name = f"TradeDecisionSchema_{feature_names}"

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
    def get_schema_for_features(features: Set[Feature]) -> Type[BaseModel]:
        """
        Convenience method to get a schema for a set of features.

        Args:
            features: Set of enabled Feature enums

        Returns:
            A Pydantic BaseModel class with the appropriate fields
        """
        key = FeatureRegistry._create_feature_set_key(features)
        return FeatureRegistry.create_trade_decision_model(key)

    @staticmethod
    def get_all_features() -> Set[Feature]:
        """Get all available features (for backward compatibility mode)"""
        return {Feature.MEMORY, Feature.SOCIAL}

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

        return features


# Backward compatibility: Export a default schema with all features enabled
# This allows existing code to continue working without changes
TradeDecisionSchema = FeatureRegistry.get_schema_for_features(FeatureRegistry.get_all_features())
