from dataclasses import dataclass
from .information_types import InformationType, InformationSignal, SignalCategory, DEFAULT_CAPABILITIES

@dataclass
class ProviderConfig:
    """Base configuration for providers"""
    category: SignalCategory = SignalCategory.PUBLIC  # Default to PUBLIC
    reliability: float = 1.0
    noise_level: float = 0.0
    update_frequency: int = 1

    def __post_init__(self):
        # Set defaults based on category
        defaults = DEFAULT_CAPABILITIES[self.category]
        if self.reliability == 1.0:
            self.reliability = defaults.accuracy
        if self.noise_level == 0.0:
            self.noise_level = defaults.noise_level

class BaseProvider:
    def __init__(self, market_state_manager, config: ProviderConfig = ProviderConfig()):
        self._market_state_manager = market_state_manager
        self.config = config
    
    @property
    def market_state(self):
        """Get current market state"""
        return self._market_state_manager.get_observable_state()

class MarketPriceProvider(BaseProvider):
    def generate_signal(self, round_number: int) -> InformationSignal:
        """Get current market price information"""
        state = self.market_state
        market_state = state['market']
        metadata = state['metadata']
        
        return InformationSignal(
            type=InformationType.PRICE,
            value=market_state['price'],
            reliability=self.config.reliability,
            metadata={
                'round': metadata['round'],  # Get round from metadata
                'best_bid': market_state['best_bid'],
                'best_ask': market_state['best_ask']
            }
        )

class InterestProvider(BaseProvider):
    def generate_signal(self, round_number: int) -> InformationSignal:
        state = self.market_state
        interest_state = state['interest']
        
        return InformationSignal(
            type=InformationType.INTEREST,
            value=interest_state['rate'],  # Current interest rate
            reliability=self.config.reliability,
            metadata={
                'round': round_number,
                'compound_frequency': interest_state['compound_frequency'],
                'last_payment': interest_state.get('last_payment'),
                'next_payment_round': interest_state.get('next_payment_round'),
                'interest_destination': interest_state.get('destination', 'dividend')  # Add destination from model
            }
        )


class BorrowRateProvider(BaseProvider):
    def generate_signal(self, round_number: int) -> InformationSignal:
        state = self.market_state
        borrow_state = state['borrow']

        return InformationSignal(
            type=InformationType.BORROW_FEE,
            value=borrow_state['rate'],
            reliability=self.config.reliability,
            metadata={
                'round': round_number,
                'payment_frequency': borrow_state['payment_frequency'],
                'last_payment': borrow_state.get('last_payment'),
                'next_payment_round': borrow_state.get('next_payment_round')
            }
        )

class OrderBookProvider(BaseProvider):
    def generate_signal(self, round_number: int) -> InformationSignal:
        state = self.market_state
        market_state = state['market']
        return InformationSignal(
            type=InformationType.ORDER_BOOK,
            value=market_state['order_book'],
            reliability=self.config.reliability,
            metadata={
                'best_bid': market_state['best_bid'],
                'best_ask': market_state['best_ask'],
                'depth_levels': len(market_state['order_book']['buy_levels']) + 
                              len(market_state['order_book']['sell_levels'])
            }
        )

class FundamentalProvider(BaseProvider):
    def generate_signal(self, round_number: int) -> InformationSignal:
        state = self.market_state
        fundamental_state = state['fundamental']  # Get fundamental section
        return InformationSignal(
            type=InformationType.FUNDAMENTAL,
            value=fundamental_state['price'],  # Access price from fundamental state
            reliability=self.config.reliability,
            metadata={
                'round': round_number,
                'periods_remaining': fundamental_state['periods_remaining'],  # Access from fundamental state
                'redemption_value': fundamental_state['redemption_value']  # Include redemption value
            }
        )

class DividendProvider(BaseProvider):
    def generate_signal(self, round_number: int) -> InformationSignal:
        """Get dividend information with yields calculated here"""
        if not self._market_state_manager.dividend_service:
            return None
            
        state = self.market_state
        dividend_state = state['dividend']
        current_price = self._market_state_manager.current_price
        model = dividend_state['model']
        
        return InformationSignal(
            type=InformationType.DIVIDEND,
            value=model.expected_dividend,
            reliability=self.config.reliability,
            metadata={
                'yields': {
                    'expected': self._calculate_yield(model.expected_dividend, current_price),
                    'max': self._calculate_yield(model.max_dividend, current_price),
                    'min': self._calculate_yield(model.min_dividend, current_price),
                    'last': self._calculate_yield(dividend_state['last_paid_dividend'], current_price) 
                           if dividend_state['last_paid_dividend'] else None
                },
                'max_dividend': model.max_dividend,
                'min_dividend': model.min_dividend,
                'last_paid_dividend': dividend_state['last_paid_dividend'],
                'next_payment_round': dividend_state['next_payment_round'],
                'should_pay': dividend_state['should_pay'],
                'variation': model.dividend_variation,
                'probability': model.dividend_probability * 100,  # Convert to percentage
                'destination': dividend_state.get('destination', 'cash'),
                'tradeable': dividend_state.get('tradeable', 'non-tradeable')
            }
        )
        
    def _calculate_yield(self, dividend: float, price: float) -> float:
        """Calculate yield percentage"""
        if price <= 0 or dividend is None:
            return 0
        return (dividend / price) * 100

class VolumeProvider(BaseProvider):
    def generate_signal(self, round_number: int) -> InformationSignal:
        state = self.market_state
        return InformationSignal(
            type=InformationType.VOLUME,
            value=state['market']['volume'],
            reliability=self.config.reliability,
            metadata={
                'round': round_number,
                'trade_history': state['market']['trade_history']
            }
        )