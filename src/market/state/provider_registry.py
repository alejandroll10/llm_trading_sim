"""Provider registry for managing information provider configuration and registration"""
from market.information.information_types import InformationType
from market.information.information_providers import (
    ProviderConfig,
    MarketPriceProvider,
    OrderBookProvider,
    FundamentalProvider,
    DividendProvider,
    InterestProvider,
    VolumeProvider,
    BorrowRateProvider,
    NewsProvider
)
from services.logging_service import LoggingService


class ProviderRegistry:
    """Manages information provider registration and configuration"""

    @staticmethod
    def register_providers(information_service, market_state_manager,
                          dividend_service=None, interest_service=None,
                          borrow_service=None, hide_fundamental_price=False,
                          news_enabled=False, news_service=None, total_rounds=20):
        """Register all information providers with the information service

        Args:
            information_service: Service to register providers with
            market_state_manager: Manager instance for provider dependencies
            dividend_service: Optional dividend service
            interest_service: Optional interest service
            borrow_service: Optional borrow service
            hide_fundamental_price: Whether to hide fundamental price
        """
        if information_service is None:
            raise RuntimeError("Information service not initialized")

        # Define provider configurations
        base_config = ProviderConfig(
            reliability=1.0,
            noise_level=0.0,
            update_frequency=1
        )

        # Core market information providers (high reliability)
        providers = {
            InformationType.PRICE: MarketPriceProvider(
                market_state_manager=market_state_manager,
                config=base_config
            ),
            InformationType.VOLUME: VolumeProvider(
                market_state_manager=market_state_manager,
                config=base_config
            ),
            InformationType.ORDER_BOOK: OrderBookProvider(
                market_state_manager=market_state_manager,
                config=base_config
            ),

            # Fundamental information (potentially noisy)
            InformationType.FUNDAMENTAL: FundamentalProvider(
                market_state_manager=market_state_manager,
                config=ProviderRegistry._get_fundamental_config(hide_fundamental_price)
            ),
        }

        # Optional providers based on available services
        if dividend_service:
            providers[InformationType.DIVIDEND] = DividendProvider(
                market_state_manager=market_state_manager,
                config=base_config
            )

        if interest_service:
            providers[InformationType.INTEREST] = InterestProvider(
                market_state_manager=market_state_manager,
                config=base_config
            )

        if borrow_service:
            providers[InformationType.BORROW_FEE] = BorrowRateProvider(
                market_state_manager=market_state_manager,
                config=base_config
            )

        # News provider (LLM-generated market news)
        if news_enabled:
            providers[InformationType.NEWS] = NewsProvider(
                market_state_manager=market_state_manager,
                config=base_config,
                news_service=news_service,
                total_rounds=total_rounds
            )

        # Register all providers
        for info_type, provider in providers.items():
            LoggingService.get_logger('simulation').debug(
                f"Registering provider for {info_type.value}"
            )
            information_service.register_provider(info_type, provider)

    @staticmethod
    def _get_fundamental_config(hide_fundamental_price: bool) -> ProviderConfig:
        """Get configuration for fundamental price provider

        Args:
            hide_fundamental_price: Whether to hide/obscure fundamental price

        Returns:
            ProviderConfig with appropriate settings
        """
        return ProviderConfig(
            reliability=0.9 if not hide_fundamental_price else 1.0,
            noise_level=0.1 if not hide_fundamental_price else 0.0,
            update_frequency=1
        )
