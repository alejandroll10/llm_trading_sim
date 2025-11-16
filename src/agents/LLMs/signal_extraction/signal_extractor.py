"""Signal extraction logic for preparing data from market information signals"""
from typing import Dict, Any
from market.information.information_types import InformationSignal


class SignalExtractor:
    """Extracts and prepares data from market information signals"""

    @staticmethod
    def extract_dividend_context(dividend_signal: InformationSignal) -> Dict[str, Any]:
        """Extract dividend context from signal

        Args:
            dividend_signal: Dividend information signal

        Returns:
            Dictionary containing dividend-related context data
        """
        metadata = dividend_signal.metadata
        yields = metadata['yields']
        model = metadata.get('model', {})

        return {
            # Current status
            'expected_dividend': dividend_signal.value,
            'expected_yield': yields['expected'],
            'last_paid_text': (
                f"${metadata['last_paid_dividend']:.2f}"
                if metadata.get('last_paid_dividend') is not None
                else "No dividends paid yet"
            ),

            # Dividend model details
            'base_dividend': dividend_signal.value,  # Use expected as base if not provided
            'variation': metadata.get('variation', 0.0),
            'max_dividend': metadata.get('max_dividend', 0.0),
            'min_dividend': metadata.get('min_dividend', 0.0),
            'max_yield': yields['max'],
            'min_yield': yields['min'],
            'probability_percent': metadata.get('probability', 50.0),
            'inverse_probability_percent': metadata.get('probability', 50.0),

            # Payment info
            'next_payment_round': metadata['next_payment_round'],
            'should_pay': metadata['should_pay'],
            'dividend_destination': metadata.get('destination', 'dividend'),
            'dividend_tradeable': metadata.get('tradeable', 'non-tradeable')
        }

    @staticmethod
    def extract_interest_context(interest_signal: InformationSignal) -> Dict[str, Any]:
        """Extract interest context from signal

        Args:
            interest_signal: Interest information signal

        Returns:
            Dictionary containing interest-related context data
        """
        metadata = interest_signal.metadata
        interest_destination = metadata.get('interest_destination', 'main')

        return {
            'interest_rate': interest_signal.value * 100,  # Convert to percentage
            'compound_frequency': metadata['compound_frequency'],
            'last_payment': metadata.get('last_payment'),
            'next_payment_round': metadata.get('next_payment_round'),
            'interest_destination': interest_destination,
            'interest_tradeable': "available for trading" if interest_destination == 'main' else "separate from trading"
        }

    @staticmethod
    def extract_redemption_context(fundamental_signal: InformationSignal) -> Dict[str, Any]:
        """Extract redemption information from fundamental signal

        Args:
            fundamental_signal: Fundamental value signal

        Returns:
            Dictionary containing redemption context
        """
        redemption_value = fundamental_signal.metadata.get('redemption_value')
        periods_remaining = fundamental_signal.metadata.get('periods_remaining')

        if periods_remaining == "Infinite" or periods_remaining is None:
            redemption_text = "This market has an infinite time horizon. Shares will not be redeemed."
        else:
            rounds_left = int(periods_remaining)
            # Check if redemption_value is None and use fundamental value instead
            if redemption_value is None:
                redemption_value = 0

            if rounds_left > 0:
                redemption_text = f"At the end of the final round (in {rounds_left} rounds), all shares will be redeemed at ${redemption_value:.2f} per share."
            else:
                redemption_text = f"This is the final round. At the end of this round, all shares will be redeemed at ${redemption_value:.2f} per share."

        return {
            'redemption_text': redemption_text
        }
