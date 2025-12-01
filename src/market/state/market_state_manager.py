from typing import Optional
from services.logging_service import LoggingService
from market.state.provider_registry import ProviderRegistry
from market.state.component_manager import ComponentManager

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
                 hide_fundamental_price=False,
                 news_enabled=False):
        self.context = context
        self.order_book = order_book
        self.agent_repository = agent_repository
        self.information_service = information_service
        self.interest_service = interest_service
        self.hide_fundamental_price = hide_fundamental_price
        self.news_enabled = news_enabled

        # Create component manager to handle updates and formatting
        self.component_manager = ComponentManager(
            context=context,
            order_book=order_book,
            dividend_service=dividend_service,
            interest_service=interest_service,
            borrow_service=borrow_service,
            hide_fundamental_price=hide_fundamental_price
        )

    @property
    def current_price(self) -> float:
        """Get current market price"""
        return self.context.current_price

    @property
    def dividend_service(self):
        """Access dividend service through component manager"""
        return self.component_manager.dividend_service

    @property
    def borrow_service(self):
        """Access borrow service through component manager"""
        return self.component_manager.borrow_service


    def update(
        self,
        round_number: int,
        last_volume: float = 0,
        is_round_end: bool = False,
        skip_distribution: bool = False,
        systematic_shock: float = 0.0,
        style_shock: float = 0.0
    ):
        LoggingService.get_logger('simulation').info(f"Updating market state for round {round_number}, is_round_end: {is_round_end}")
        # 1. Update all market state components first
        self._update_market_components(round_number, last_volume)

        # 2. Then distribute information (unless explicitly skipped for multi-stock)
        if not is_round_end and not skip_distribution:
            self._distribute_market_information(round_number)

        # 3. Handle end-of-round if needed (pass shocks for dividend calculation)
        if is_round_end:
            self._process_end_of_round(
                round_number,
                systematic_shock=systematic_shock,
                style_shock=style_shock
            )

        # 4. Return final state
        return self._get_current_market_state(round_number, last_volume)

    def _update_market_components(self, round_number: int, last_volume: float):
        """Update all market components in the correct order"""
        self.component_manager.update_components(round_number)

    def _distribute_market_information(self, round_number: int):
        """Handle all information distribution"""
        if not self.information_service:
            LoggingService.get_logger('simulation').error("No information service found")
            raise RuntimeError("No information service found")

        # Ensure providers are registered
        if not self.information_service.providers:
            ProviderRegistry.register_providers(
                information_service=self.information_service,
                market_state_manager=self,
                dividend_service=self.component_manager.dividend_service,
                interest_service=self.interest_service,
                borrow_service=self.component_manager.borrow_service,
                hide_fundamental_price=self.hide_fundamental_price,
                news_enabled=self.news_enabled,
                total_rounds=self.context._num_rounds
            )

        # Distribute information
        self.information_service.distribute_information(round_number)

    def _process_end_of_round(
        self,
        round_number: int,
        systematic_shock: float = 0.0,
        style_shock: float = 0.0
    ):
        """Handle end-of-round payments and processing.

        Args:
            round_number: Current simulation round
            systematic_shock: Market-wide dividend shock (same for all stocks)
            style_shock: Style-level dividend shock (same for stocks in same style)
        """
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
        if self.component_manager.borrow_service:
            borrow_result = self.component_manager.borrow_service.process_borrow_fees(
                round_number, self.context.current_price
            )
            if borrow_result.total_fee > 0:
                self.context.record_borrow_fee_payment(borrow_result.total_fee, round_number)

        # 3. Process dividend payments if needed (with shock components)
        if self.component_manager.dividend_service:
            LoggingService.get_logger('simulation').info(f"Processing dividend payments for round {round_number}")
            payment_result = self.component_manager.dividend_service.process_round_end(
                round_number,
                is_final_round,
                systematic_shock=systematic_shock,
                style_shock=style_shock
            )
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
        """Access dividend model through component manager"""
        return self.component_manager.dividend_model

    def update_market_depth(self):
        """Update market depth information from order book"""
        self.component_manager.update_market_depth()
    
    def _get_current_market_state(self, round_number: int, last_volume: float) -> dict:
        """Get current market state"""
        return self.component_manager.get_current_market_state(round_number, last_volume)

    def get_observable_state(self):
        """Single source of truth for market state"""
        return self.component_manager.format_observable_state()

    def _format_market_state(self, public_info: dict) -> dict:
        """Format market component state"""
        return self.component_manager.format_market_state(public_info)

    def _format_fundamental_state(self) -> dict:
        """Format fundamental component state"""
        return self.component_manager.format_fundamental_state()

    def _format_dividend_state(self) -> Optional[dict]:
        """Format dividend component state"""
        return self.component_manager.format_dividend_state()

    def _format_interest_state(self) -> dict:
        """Format interest state information"""
        return self.component_manager.format_interest_state(self.interest_service)

    def _format_borrow_state(self) -> dict:
        """Format borrow fee state information"""
        return self.component_manager.format_borrow_state()

    def _format_metadata(self, public_info: dict) -> dict:
        """Format metadata component"""
        return self.component_manager.format_metadata(public_info)
    