from dataclasses import dataclass
from typing import Dict, Any
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

    def generate_signal_for_manager(self, manager, round_number: int) -> InformationSignal:
        """Generate signal using specific manager (multi-stock support)"""
        state = manager.get_observable_state()
        market_state = state['market']
        metadata = state['metadata']

        return InformationSignal(
            type=InformationType.PRICE,
            value=market_state['price'],
            reliability=self.config.reliability,
            metadata={
                'round': metadata['round'],
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

    def generate_signal_for_manager(self, manager, round_number: int) -> InformationSignal:
        """Generate signal using specific manager (multi-stock support)"""
        state = manager.get_observable_state()
        interest_state = state['interest']

        return InformationSignal(
            type=InformationType.INTEREST,
            value=interest_state['rate'],
            reliability=self.config.reliability,
            metadata={
                'round': round_number,
                'compound_frequency': interest_state['compound_frequency'],
                'last_payment': interest_state.get('last_payment'),
                'next_payment_round': interest_state.get('next_payment_round'),
                'interest_destination': interest_state.get('destination', 'dividend')
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

    def generate_signal_for_manager(self, manager, round_number: int) -> InformationSignal:
        """Generate signal using specific manager (multi-stock support)"""
        state = manager.get_observable_state()
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

    def generate_signal_for_manager(self, manager, round_number: int) -> InformationSignal:
        """Generate signal using specific manager (multi-stock support)"""
        state = manager.get_observable_state()
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

    def generate_signal_for_manager(self, manager, round_number: int) -> InformationSignal:
        """Generate signal using specific manager (multi-stock support)"""
        state = manager.get_observable_state()
        fundamental_state = state['fundamental']
        return InformationSignal(
            type=InformationType.FUNDAMENTAL,
            value=fundamental_state['price'],
            reliability=self.config.reliability,
            metadata={
                'round': round_number,
                'periods_remaining': fundamental_state['periods_remaining'],
                'redemption_value': fundamental_state['redemption_value']
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

    def generate_signal_for_manager(self, manager, round_number: int) -> InformationSignal:
        """Generate signal using specific manager (multi-stock support)"""
        if not manager.dividend_service:
            return None

        state = manager.get_observable_state()
        dividend_state = state['dividend']
        current_price = manager.current_price
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
                'probability': model.dividend_probability * 100,
                'destination': dividend_state.get('destination', 'cash'),
                'tradeable': dividend_state.get('tradeable', 'non-tradeable')
            }
        )

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

    def generate_signal_for_manager(self, manager, round_number: int) -> InformationSignal:
        """Generate signal using specific manager (multi-stock support)"""
        state = manager.get_observable_state()
        return InformationSignal(
            type=InformationType.VOLUME,
            value=state['market']['volume'],
            reliability=self.config.reliability,
            metadata={
                'round': round_number,
                'trade_history': state['market']['trade_history']
            }
        )


class NewsProvider(BaseProvider):
    """
    Provider for LLM-generated market news.

    Unlike other providers, this one actively generates content via LLM
    rather than just reading market state. The news service handles the
    actual LLM call and ensures only PUBLIC information is used.

    For multi-stock scenarios, news is generated ONCE for all stocks
    (single LLM call) and cached. The returned news may include:
    - Market-wide news (affected_stocks=None)
    - Stock-specific news (affected_stocks=["STOCK_A"])
    - Multi-stock news (affected_stocks=["STOCK_A", "STOCK_B"])
    """

    # Class-level cache for multi-stock scenarios (shared across instances)
    _multi_stock_cache: Dict[int, list] = {}

    def __init__(self, market_state_manager, config: ProviderConfig = ProviderConfig(),
                 news_service=None, total_rounds: int = 20):
        super().__init__(market_state_manager, config)
        self.total_rounds = total_rounds
        self._news_service = news_service
        self._price_history = []  # Track prices for context
        self._news_cache = {}     # Cache: {round_number: [NewsItem, ...]} for single-stock

    @property
    def news_service(self):
        """Lazy initialization of news service"""
        if self._news_service is None:
            from services.news_service import NewsService
            self._news_service = NewsService()
        return self._news_service

    def _update_price_history(self):
        """Update price history from market state"""
        try:
            state = self.market_state
            current_price = state.get('market', {}).get('price')
            if current_price and (not self._price_history or
                                  self._price_history[-1] != current_price):
                self._price_history.append(current_price)
                # Keep last 10 prices
                if len(self._price_history) > 10:
                    self._price_history = self._price_history[-10:]
        except Exception:
            pass  # Ignore price history errors

    def generate_signal(self, round_number: int) -> InformationSignal:
        """Get news as an InformationSignal (single-stock mode)"""
        if round_number not in self._news_cache:
            self._update_price_history()
            state = self.market_state
            news_items = self.news_service.generate_news(
                round_number=round_number,
                total_rounds=self.total_rounds,
                market_state=state,
                price_history=self._price_history,
                stock_id=None
            )
            self._news_cache[round_number] = news_items

        news_items = self._news_cache[round_number]

        return InformationSignal(
            type=InformationType.NEWS,
            value=news_items,
            reliability=self.config.reliability,
            metadata={
                'round': round_number,
                'item_count': len(news_items),
                'sentiments': [item.sentiment for item in news_items] if news_items else [],
            }
        )

    def generate_news_for_all_stocks(self, round_number: int, managers: Dict[str, Any]) -> list:
        """
        Generate news for all stocks in one LLM call (multi-stock mode).

        Args:
            round_number: Current round
            managers: Dict of {stock_id: market_state_manager}

        Returns:
            List of NewsItem objects (cached at class level)
        """
        if round_number not in NewsProvider._multi_stock_cache:
            # Collect market state from all managers
            stocks_data = {}
            for stock_id, manager in managers.items():
                stocks_data[stock_id] = manager.get_observable_state()

            # Single LLM call for all stocks
            news_items = self.news_service.generate_news_multi_stock(
                round_number=round_number,
                total_rounds=self.total_rounds,
                stocks_data=stocks_data
            )
            NewsProvider._multi_stock_cache[round_number] = news_items

        return NewsProvider._multi_stock_cache[round_number]

    def generate_signal_for_manager(self, manager, round_number: int) -> InformationSignal:
        """
        Generate signal for a specific market manager (multi-stock).

        In multi-stock mode, this returns ALL news (market-wide + stock-specific).
        Agents see all news and can decide what's relevant.
        """
        stock_id = getattr(manager, 'stock_id', None)

        # Check if we have multi-stock cached news for this round
        if round_number in NewsProvider._multi_stock_cache:
            news_items = NewsProvider._multi_stock_cache[round_number]
        else:
            # Fallback: generate for single stock (shouldn't happen in proper multi-stock flow)
            state = manager.get_observable_state()
            news_items = self.news_service.generate_news(
                round_number=round_number,
                total_rounds=self.total_rounds,
                market_state=state,
                price_history=self._price_history,
                stock_id=stock_id
            )

        return InformationSignal(
            type=InformationType.NEWS,
            value=news_items,
            reliability=self.config.reliability,
            metadata={
                'round': round_number,
                'stock_id': stock_id,
                'item_count': len(news_items),
                'sentiments': [item.sentiment for item in news_items] if news_items else [],
            }
        )

    def clear_cache(self):
        """Clear news cache (call between simulations if reusing provider)"""
        self._news_cache.clear()
        self._price_history.clear()
        NewsProvider._multi_stock_cache.clear()