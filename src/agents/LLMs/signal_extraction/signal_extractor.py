"""Signal extraction logic for preparing data from market information signals"""
from typing import Dict, Any
from market.information.information_types import InformationSignal
from scenarios.base import FundamentalInfoMode


class SignalExtractor:
    """Extracts and prepares data from market information signals"""

    @staticmethod
    def extract_dividend_context(
        dividend_signal: InformationSignal,
        mode: FundamentalInfoMode = FundamentalInfoMode.FULL
    ) -> Dict[str, Any]:
        """Extract dividend context from signal based on information mode.

        Args:
            dividend_signal: Dividend information signal (includes dividend_history in metadata)
            mode: Controls what information to reveal (FULL, PROCESS_ONLY, etc.)

        Returns:
            Dictionary containing dividend-related context data
        """
        metadata = dividend_signal.metadata
        yields = metadata['yields']

        # Get dividend history from signal metadata (set by DividendProvider)
        dividend_history = metadata.get('dividend_history', [])

        # NONE mode: no dividend information at all
        if mode == FundamentalInfoMode.NONE:
            return {
                'dividend_info_available': False,
                'dividend_info_mode': 'none'
            }

        # REALIZATIONS_ONLY mode: only show past dividend payments
        if mode == FundamentalInfoMode.REALIZATIONS_ONLY:
            return {
                'dividend_info_available': True,
                'dividend_info_mode': 'realizations_only',
                # Only past payments, no model
                'last_paid_text': (
                    f"${metadata['last_paid_dividend']:.2f}"
                    if metadata.get('last_paid_dividend') is not None
                    else "No dividends paid yet"
                ),
                'dividend_history': dividend_history,
                'num_payments': len(dividend_history),
                # Payment schedule (when, not how much)
                'next_payment_round': metadata['next_payment_round'],
                'dividend_destination': metadata.get('destination', 'dividend'),
                'dividend_tradeable': metadata.get('tradeable', 'non-tradeable')
            }

        # AVERAGE mode: show running statistics of past dividends
        if mode == FundamentalInfoMode.AVERAGE:
            if dividend_history:
                import statistics
                avg = statistics.mean(dividend_history)
                std = statistics.stdev(dividend_history) if len(dividend_history) > 1 else 0.0
            else:
                avg = None
                std = None

            return {
                'dividend_info_available': True,
                'dividend_info_mode': 'average',
                'last_paid_text': (
                    f"${metadata['last_paid_dividend']:.2f}"
                    if metadata.get('last_paid_dividend') is not None
                    else "No dividends paid yet"
                ),
                'dividend_average': avg,
                'dividend_std': std,
                'num_payments': len(dividend_history),
                # Payment schedule
                'next_payment_round': metadata['next_payment_round'],
                'dividend_destination': metadata.get('destination', 'dividend'),
                'dividend_tradeable': metadata.get('tradeable', 'non-tradeable')
            }

        # PROCESS_ONLY and FULL modes: show full dividend model
        # (PROCESS_ONLY hides redemption separately, dividend model is shown)
        return {
            'dividend_info_available': True,
            'dividend_info_mode': mode.value,
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
    def extract_redemption_context(
        fundamental_signal: InformationSignal,
        mode: FundamentalInfoMode = FundamentalInfoMode.FULL
    ) -> Dict[str, Any]:
        """Extract redemption information from fundamental signal based on mode.

        Args:
            fundamental_signal: Fundamental value signal
            mode: Controls what information to reveal

        Returns:
            Dictionary containing redemption context
        """
        periods_remaining = fundamental_signal.metadata.get('periods_remaining')

        # For NONE, REALIZATIONS_ONLY, and AVERAGE modes: hide redemption value
        # This prevents agents from directly computing FV
        if mode in (FundamentalInfoMode.NONE, FundamentalInfoMode.REALIZATIONS_ONLY,
                    FundamentalInfoMode.AVERAGE):
            if periods_remaining == "Infinite" or periods_remaining is None:
                redemption_text = "This market has an infinite time horizon. Shares will not be redeemed."
            else:
                rounds_left = int(periods_remaining)
                if rounds_left > 0:
                    redemption_text = f"At the end of the final round (in {rounds_left} rounds), all shares will be redeemed. The redemption value is not disclosed."
                else:
                    redemption_text = "This is the final round. At the end of this round, all shares will be redeemed. The redemption value is not disclosed."
            return {'redemption_text': redemption_text}

        # For PROCESS_ONLY mode: also hide redemption to prevent direct FV calculation
        # (even though dividend model is shown)
        if mode == FundamentalInfoMode.PROCESS_ONLY:
            redemption_value = fundamental_signal.metadata.get('redemption_value')
            if periods_remaining == "Infinite" or periods_remaining is None:
                redemption_text = "This market has an infinite time horizon. Shares will not be redeemed."
            else:
                rounds_left = int(periods_remaining)
                if rounds_left > 0:
                    # Clarify that asset pays forever but sim ends - triggers perpetuity valuation
                    redemption_text = f"The asset pays dividends indefinitely. The simulation ends in {rounds_left} rounds, at which point shares are liquidated at fair value."
                else:
                    redemption_text = "The asset pays dividends indefinitely. This is the final round, at which point shares are liquidated at fair value."
            return {'redemption_text': redemption_text}

        # FULL mode: show everything including redemption value
        redemption_value = fundamental_signal.metadata.get('redemption_value')
        if periods_remaining == "Infinite" or periods_remaining is None:
            redemption_text = "This market has an infinite time horizon. Shares will not be redeemed."
        else:
            rounds_left = int(periods_remaining)
            if redemption_value is None:
                redemption_value = 0

            if rounds_left > 0:
                redemption_text = f"At the end of the final round (in {rounds_left} rounds), all shares will be redeemed at ${redemption_value:.2f} per share."
            else:
                redemption_text = f"This is the final round. At the end of this round, all shares will be redeemed at ${redemption_value:.2f} per share."

        return {'redemption_text': redemption_text}
