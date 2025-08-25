from datetime import datetime
from market.information.information_types import InformationType
from market.information.information_providers import ProviderConfig,MarketPriceProvider, OrderBookProvider, FundamentalProvider, DividendProvider, InterestProvider, VolumeProvider, BorrowRateProvider
from typing import Optional
from services.logging_service import LoggingService

class MarketStateManager:
    """
    Manages the dynamic state of the market.
    
    Responsibilities:
    - Updates market state each round
    - Coordinates information distribution
    - Manages market depth updates
    - Handles end-of-round processing
    
    Does NOT:
    - Store market truth (SimContext)
    - Generate information signals (InformationService)
    - Manage agent states (AgentRepository)
    - Update market truth (BaseSim)
    """

    def __init__(self, context, order_book, agent_repository, logger,  information_service,
                 dividend_service, interest_service,
                 borrow_service=None,
                 hide_fundamental_price=False):
        self.context = context
        self.order_book = order_book
        self.agent_repository = agent_repository
        self.information_service = information_service
        self.hide_fundamental_price = hide_fundamental_price
        self.dividend_service = dividend_service
        self.interest_service = interest_service
        self.borrow_service = borrow_service

    @property
    def current_price(self) -> float:
        """Get current market price"""
        return self.context.current_price


    def update(self, round_number: int, last_volume: float = 0, is_round_end: bool = False):
        LoggingService.get_logger('simulation').info(f"Updating market state for round {round_number}, is_round_end: {is_round_end}")
        # 1. Update all market state components first
        self._update_market_components(round_number, last_volume)
        
        # 2. Then distribute information
        if not is_round_end:
            self._distribute_market_information(round_number)
        
        # 3. Handle end-of-round if needed
        if is_round_end:
            self._process_end_of_round(round_number)
        
        # 4. Return final state
        return self._get_current_market_state(round_number, last_volume)

    def _update_market_components(self, round_number: int, last_volume: float):
        """Update all market components in the correct order"""
        # 1. Update order book and depth
        self.update_market_depth()
        
        # 2. Update dividend state if exists
        if self.dividend_service:
            self.dividend_service.update(round_number)
        else:
            LoggingService.get_logger('simulation').error("No dividend service found")
            raise RuntimeError("No dividend service found")

        # 3. Update borrow state if exists
        if self.borrow_service:
            self.borrow_service.update(round_number)

    def _distribute_market_information(self, round_number: int):
        """Handle all information distribution"""
        if not self.information_service:
            LoggingService.get_logger('simulation').error("No information service found")
            raise RuntimeError("No information service found")
        
        # Ensure providers are registered
        if not self.information_service.providers:
            self._register_base_providers()
        
        # Distribute information
        self.information_service.distribute_information(round_number)

    def _process_end_of_round(self, round_number: int):
        """Handle end-of-round payments and processing"""
        LoggingService.get_logger('simulation').info(f"Processing end-of-round payments for round {round_number}")
        is_final_round = round_number == self.context._num_rounds - 1 and not self.context.infinite_rounds
        if is_final_round:
            LoggingService.get_logger('simulation').info(f"Processing final round {round_number}")
        # 1. Process interest payments
        if self.interest_service:
            interest_result = self.interest_service.process_interest_payments(round_number)
            if interest_result.success:
                self.context.record_interest_payment(
                    amount=interest_result.total_payment,
                    round_number=round_number
                )

        # 2. Process borrow fee payments
        if self.borrow_service:
            self.borrow_service.process_borrow_fees(round_number, self.context.current_price)
        
        # 3. Process dividend payments if needed
        if self.dividend_service:
            LoggingService.get_logger('simulation').info(f"Processing dividend payments for round {round_number}")
            payment_result = self.dividend_service.process_round_end(round_number, is_final_round)
            LoggingService.get_logger('simulation').info(f"Payment result: {payment_result}")
            if payment_result:  # Only process if payments were made
                LoggingService.get_logger('simulation').info(
                    f"Processed dividend payments: "
                    f"${payment_result.total_payment:.2f} "
                    f"across {payment_result.num_shares_paid} shares"
                )
                self.context.record_dividend_payment(
                    amount=payment_result.total_payment,
                    round_number=round_number
                )

    @property
    def dividend_model(self):
        """Access dividend model through dividend service"""
        if self.dividend_service is None:
            return None
        # Use get_state() to access model info
        return self.dividend_service.get_state()['model']

    def update_market_depth(self):
        """Update market depth information from order book"""
        if self.order_book is None:
            raise RuntimeError("Order book not registered")
        
        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()
        midpoint = self.order_book.get_midpoint()
        
        LoggingService.log_order_state(
            f"Updating market depth - "
            f"Best bid: {best_bid if best_bid else None}, "
            f"Best ask: {best_ask if best_ask else None}, "
            f"Midpoint: {midpoint if midpoint else None}"
        )
        
        # Store historical quote
        public_info = self.context.get_public_info()
        
        # Ensure historical_quotes exists
        if 'historical_quotes' not in public_info['order_book_state']:
            public_info['order_book_state']['historical_quotes'] = []
        
        # Store historical quote
        public_info['order_book_state']['historical_quotes'].append({
            'round': self.context.round_number,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'midpoint': midpoint,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update order book state
        public_info['order_book_state'].update({
            'best_bid': best_bid,
            'best_ask': best_ask,
            'midpoint': midpoint,
            'aggregated_levels': self.order_book.get_aggregated_levels()
        })
        
        # Update context with new public info
        self.context.public_info.update(public_info)
    
    def _get_current_market_state(self, round_number: int, last_volume: float) -> dict:
        """Get current market state"""
        public_info = self.context.get_public_info()
        
        # Get dividend state safely
        dividend_state = self._format_dividend_state()
        last_paid_dividend = (
            dividend_state['last_paid_dividend'] 
            if dividend_state and 'last_paid_dividend' in dividend_state 
            else 0.0
        )
        
        return {
            'price': self.context.current_price,
            'fundamental_price': self.context.fundamental_price,
            'market_depth': self.order_book.get_aggregated_levels(),
            'best_bid': public_info['order_book_state']['best_bid'],
            'best_ask': public_info['order_book_state']['best_ask'],
            'midpoint': public_info['order_book_state']['midpoint'],
            'last_trade_price': public_info['last_trade'],
            'volume': last_volume,
            'round_number': round_number + 1,
            'num_rounds': self.context._num_rounds,
            'periods_remaining': self.context._num_rounds - round_number if not self.context.infinite_rounds else "Infinite",
            'dividend_state': dividend_state,
            'last_paid_dividend': last_paid_dividend,
            'infinite_rounds': self.context.infinite_rounds
        }

    def get_observable_state(self):
        """Single source of truth for market state"""
        public_info = self.context.get_public_info()
        
        return {
            'market': self._format_market_state(public_info),
            'fundamental': self._format_fundamental_state(),
            'dividend': self._format_dividend_state(),
            'interest': self._format_interest_state(),
            'borrow': self._format_borrow_state(),
            'metadata': self._format_metadata(public_info)
        }

    def _format_market_state(self, public_info: dict) -> dict:
        """Format market component state"""
        return {
            'price': self.context.current_price,
            'order_book': self.order_book.get_aggregated_levels(),
            'best_bid': public_info['order_book_state']['best_bid'],
            'best_ask': public_info['order_book_state']['best_ask'],
            'midpoint': public_info['order_book_state']['midpoint'],
            'last_trade_price': public_info['last_trade']['price'],
            'volume': public_info['last_trade']['volume'],
            'trade_history': public_info['trade_history'][-5:]
        }

    def _format_fundamental_state(self) -> dict:
        """Format fundamental component state"""
        return {
            'price': self.context.fundamental_price if not self.hide_fundamental_price else None,
            'periods_remaining': self.context._num_rounds - self.context.round_number if not self.context.infinite_rounds else "Infinite",
            'redemption_value': self.context.redemption_value if not self.context.infinite_rounds else None
        }

    def _format_dividend_state(self) -> Optional[dict]:
        """Format dividend component state"""
        if not self.dividend_service:
            return None
        return self.dividend_service.get_state()

    def _format_interest_state(self) -> dict:
        """Format interest state information"""
        if not self.interest_service:
            return {}
        
        return {
            'rate': self.interest_service.get_current_rate(),
            'compound_frequency': self.interest_service.interest_model['compound_frequency'],
            'last_payment': (self.interest_service.interest_history[-1] 
                            if self.interest_service.interest_history else None),
            'next_payment_round': self.interest_service.next_payment_round,
            'destination': self.interest_service.interest_model['destination']
        }

    def _format_borrow_state(self) -> dict:
        """Format borrow fee state information"""
        if not self.borrow_service:
            return {}

        return {
            'rate': self.borrow_service.get_current_rate(),
            'payment_frequency': self.borrow_service.borrow_model.get('payment_frequency', 1),
            'last_payment': (self.borrow_service.borrow_history[-1]
                             if self.borrow_service.borrow_history else None),
            'next_payment_round': self.borrow_service.next_payment_round
        }

    def _format_metadata(self, public_info: dict) -> dict:
        """Format metadata component"""
        return {
            'round': self.context.round_number,
            'last_trade': public_info['last_trade']
        }
    
    def _register_base_providers(self):
        """Register standard information providers with consistent configuration"""
        if self.information_service is None:
            raise RuntimeError("Information service not initialized")
        
        # Define provider configurations
        base_config = ProviderConfig(
            reliability=1.0,
            noise_level=0.0,
            update_frequency=1
        )
        
        noisy_config = ProviderConfig(
            reliability=0.9,
            noise_level=0.1,
            update_frequency=1
        )
        
        # Define all available providers with their configs
        providers = {
            # Core market information (high reliability)
            InformationType.PRICE: MarketPriceProvider(
                market_state_manager=self,
                config=base_config
            ),
            InformationType.VOLUME: VolumeProvider(
                market_state_manager=self,
                config=base_config
            ),
            InformationType.ORDER_BOOK: OrderBookProvider(
                market_state_manager=self,
                config=base_config
            ),
            
            # Fundamental information (potentially noisy)
            InformationType.FUNDAMENTAL: FundamentalProvider(
                market_state_manager=self,
                config=self._get_fundamental_config()
            ),
        }
        
        # Optional providers based on services
        if self.dividend_service:
            providers[InformationType.DIVIDEND] = DividendProvider(
                market_state_manager=self,
                config=base_config
            )
            
        if self.interest_service:
            providers[InformationType.INTEREST] = InterestProvider(
                market_state_manager=self,
                config=base_config
            )

        if self.borrow_service:
            providers[InformationType.BORROW_FEE] = BorrowRateProvider(
                market_state_manager=self,
                config=base_config
            )
        
        # Register all providers at once
        for info_type, provider in providers.items():
            LoggingService.get_logger('simulation').debug(f"Registering provider for {info_type.value}")
            self.information_service.register_provider(info_type, provider)

    def _get_fundamental_config(self) -> ProviderConfig:
        """Get configuration for fundamental price provider"""
        return ProviderConfig(
            reliability=0.9 if not self.hide_fundamental_price else 1.0,
            noise_level=0.1 if not self.hide_fundamental_price else 0.0,
            update_frequency=1
        )
    
    def _get_dividend_config(self) -> ProviderConfig:
        """Get configuration for dividend provider"""
        return ProviderConfig(
            reliability=0.9,  # Dividends have some uncertainty
            noise_level=0.1,  # Add some noise to expected dividends
            update_frequency=1
        )
    