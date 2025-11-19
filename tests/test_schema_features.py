"""
Unit tests for the feature toggle schema system.

Tests:
    - Feature extraction from configuration
    - Dynamic schema generation with different feature combinations
    - Schema caching behavior
    - Field validation for all combinations
"""

import pytest
from pydantic import ValidationError
from agents.LLMs.services.schema_features import (
    Feature,
    FeatureRegistry,
    OrderSchema,
    TradeDecisionSchema
)


class TestFeatureExtraction:
    """Test extracting features from configuration dictionaries"""

    def test_all_features_enabled(self):
        """Test extracting all enabled features"""
        config = {
            'MEMORY_ENABLED': True,
            'SOCIAL_ENABLED': True
        }
        features = FeatureRegistry.extract_features_from_config(config)
        assert features == {Feature.MEMORY, Feature.SOCIAL}

    def test_only_memory_enabled(self):
        """Test with only memory enabled"""
        config = {
            'MEMORY_ENABLED': True,
            'SOCIAL_ENABLED': False
        }
        features = FeatureRegistry.extract_features_from_config(config)
        assert features == {Feature.MEMORY}

    def test_only_social_enabled(self):
        """Test with only social enabled"""
        config = {
            'MEMORY_ENABLED': False,
            'SOCIAL_ENABLED': True
        }
        features = FeatureRegistry.extract_features_from_config(config)
        assert features == {Feature.SOCIAL}

    def test_no_features_enabled(self):
        """Test with all features disabled"""
        config = {
            'MEMORY_ENABLED': False,
            'SOCIAL_ENABLED': False
        }
        features = FeatureRegistry.extract_features_from_config(config)
        assert features == set()

    def test_default_behavior(self):
        """Test that features default to enabled for backward compatibility"""
        config = {}  # No feature flags specified
        features = FeatureRegistry.extract_features_from_config(config)
        assert features == {Feature.MEMORY, Feature.SOCIAL}


class TestSchemaGeneration:
    """Test dynamic schema generation based on features"""

    def test_schema_with_all_features(self):
        """Test schema generation with all features enabled"""
        features = {Feature.MEMORY, Feature.SOCIAL}
        schema_class = FeatureRegistry.get_schema_for_features(features)

        # Check that all fields are present
        assert 'valuation_reasoning' in schema_class.model_fields
        assert 'valuation' in schema_class.model_fields
        assert 'price_target_reasoning' in schema_class.model_fields
        assert 'price_target' in schema_class.model_fields
        assert 'reasoning' in schema_class.model_fields
        assert 'orders' in schema_class.model_fields
        assert 'replace_decision' in schema_class.model_fields
        assert 'notes_to_self' in schema_class.model_fields  # Memory
        assert 'message_reasoning' in schema_class.model_fields  # Social
        assert 'post_message' in schema_class.model_fields  # Social

    def test_schema_with_no_features(self):
        """Test schema generation with no features (core only)"""
        features = set()
        schema_class = FeatureRegistry.get_schema_for_features(features)

        # Check that only core fields are present
        assert 'valuation_reasoning' in schema_class.model_fields
        assert 'orders' in schema_class.model_fields
        assert 'reasoning' in schema_class.model_fields

        # Check that feature fields are NOT present
        assert 'notes_to_self' not in schema_class.model_fields
        assert 'message_reasoning' not in schema_class.model_fields
        assert 'post_message' not in schema_class.model_fields

    def test_schema_with_only_memory(self):
        """Test schema with only memory enabled"""
        features = {Feature.MEMORY}
        schema_class = FeatureRegistry.get_schema_for_features(features)

        # Memory fields should be present
        assert 'notes_to_self' in schema_class.model_fields

        # Social fields should NOT be present
        assert 'message_reasoning' not in schema_class.model_fields
        assert 'post_message' not in schema_class.model_fields

    def test_schema_with_only_social(self):
        """Test schema with only social enabled"""
        features = {Feature.SOCIAL}
        schema_class = FeatureRegistry.get_schema_for_features(features)

        # Social fields should be present
        assert 'message_reasoning' in schema_class.model_fields
        assert 'post_message' in schema_class.model_fields

        # Memory fields should NOT be present
        assert 'notes_to_self' not in schema_class.model_fields


class TestSchemaValidation:
    """Test that generated schemas validate correctly"""

    def test_valid_decision_all_features(self):
        """Test validation with all features and all fields populated"""
        features = {Feature.MEMORY, Feature.SOCIAL}
        schema_class = FeatureRegistry.get_schema_for_features(features)

        # Create a valid decision with all fields
        decision = schema_class(
            valuation_reasoning="Stock is undervalued by 10%",
            valuation=30.0,
            price_target_reasoning="Expecting price increase",
            price_target=32.0,
            reasoning="Buying because undervalued",
            orders=[{
                'decision': 'Buy',
                'quantity': 100,
                'order_type': 'Market',
                'stock_id': 'DEFAULT_STOCK'
            }],
            replace_decision="Replace",
            notes_to_self="Stock trending up",
            message_reasoning="Want to signal confidence",
            post_message="Bullish on this stock!"
        )

        assert decision.valuation == 30.0
        assert decision.notes_to_self == "Stock trending up"
        assert decision.post_message == "Bullish on this stock!"

    def test_valid_decision_no_features(self):
        """Test validation with no features (core only)"""
        features = set()
        schema_class = FeatureRegistry.get_schema_for_features(features)

        # Create a valid decision with only core fields
        decision = schema_class(
            valuation_reasoning="Stock is undervalued",
            valuation=30.0,
            price_target_reasoning="Expecting increase",
            price_target=32.0,
            reasoning="Buying",
            orders=[],
            replace_decision="Add"
        )

        assert decision.valuation == 30.0
        assert not hasattr(decision, 'notes_to_self')

    def test_optional_fields_can_be_none(self):
        """Test that optional feature fields can be None"""
        features = {Feature.MEMORY, Feature.SOCIAL}
        schema_class = FeatureRegistry.get_schema_for_features(features)

        # Create decision without optional fields
        decision = schema_class(
            valuation_reasoning="Analysis",
            valuation=30.0,
            price_target_reasoning="Prediction",
            price_target=32.0,
            reasoning="Strategy",
            orders=[],
            replace_decision="Add",
            notes_to_self=None,  # Optional
            message_reasoning=None,  # Optional
            post_message=None  # Optional
        )

        assert decision.notes_to_self is None
        assert decision.post_message is None

    def test_missing_required_field_raises_error(self):
        """Test that missing required fields raise validation errors"""
        features = {Feature.MEMORY}
        schema_class = FeatureRegistry.get_schema_for_features(features)

        # Try to create decision without required field
        with pytest.raises(ValidationError):
            schema_class(
                # Missing valuation_reasoning
                valuation=30.0,
                price_target_reasoning="Prediction",
                price_target=32.0,
                reasoning="Strategy",
                orders=[],
                replace_decision="Add"
            )


class TestSchemaCaching:
    """Test that schema caching works correctly"""

    def test_same_features_return_same_schema(self):
        """Test that requesting the same features returns cached schema"""
        features1 = {Feature.MEMORY, Feature.SOCIAL}
        features2 = {Feature.SOCIAL, Feature.MEMORY}  # Different order

        schema1 = FeatureRegistry.get_schema_for_features(features1)
        schema2 = FeatureRegistry.get_schema_for_features(features2)

        # Should be the exact same class (from cache)
        assert schema1 is schema2

    def test_different_features_return_different_schemas(self):
        """Test that different features create different schemas"""
        schema_all = FeatureRegistry.get_schema_for_features({Feature.MEMORY, Feature.SOCIAL})
        schema_memory = FeatureRegistry.get_schema_for_features({Feature.MEMORY})
        schema_none = FeatureRegistry.get_schema_for_features(set())

        # Should be different classes
        assert schema_all is not schema_memory
        assert schema_memory is not schema_none
        assert schema_all is not schema_none


class TestBackwardCompatibility:
    """Test backward compatibility with existing code"""

    def test_default_schema_has_all_features(self):
        """Test that the default exported schema has all features"""
        # The TradeDecisionSchema export should have all features for compatibility
        assert 'notes_to_self' in TradeDecisionSchema.model_fields
        assert 'message_reasoning' in TradeDecisionSchema.model_fields
        assert 'post_message' in TradeDecisionSchema.model_fields

    def test_get_all_features(self):
        """Test that get_all_features returns all available features"""
        all_features = FeatureRegistry.get_all_features()
        assert Feature.MEMORY in all_features
        assert Feature.SOCIAL in all_features


class TestOrderSchema:
    """Test that OrderSchema remains unchanged"""

    def test_order_schema_fields(self):
        """Test OrderSchema has expected fields"""
        assert 'decision' in OrderSchema.model_fields
        assert 'quantity' in OrderSchema.model_fields
        assert 'order_type' in OrderSchema.model_fields
        assert 'price_limit' in OrderSchema.model_fields
        assert 'stock_id' in OrderSchema.model_fields

    def test_order_creation(self):
        """Test creating a valid order"""
        order = OrderSchema(
            decision='Buy',
            quantity=100,
            order_type='Market',
            stock_id='TECH_A'
        )

        assert order.decision == 'Buy'
        assert order.quantity == 100
        assert order.stock_id == 'TECH_A'

    def test_order_default_stock_id(self):
        """Test that stock_id defaults to DEFAULT_STOCK"""
        order = OrderSchema(
            decision='Sell',
            quantity=50,
            order_type='Limit',
            price_limit=100.0
        )

        assert order.stock_id == 'DEFAULT_STOCK'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
