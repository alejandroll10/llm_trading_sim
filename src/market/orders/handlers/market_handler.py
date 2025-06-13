from typing import List, Tuple
from market.orders.order import Order, OrderState
from market.trade import Trade
from market.orders.handlers.base_handler import BaseOrderHandler
from market.orders.handlers.services.validation_service import OrderValidationService
from market.orders.handlers.services.matching_service import OrderMatchingService
from market.orders.handlers.services.conversion_service import OrderConversionService
from market.orders.handlers.services.sync_service import AgentSyncService
from services.logging_service import LoggingService

class MarketOrderHandler(BaseOrderHandler):
    """
    Handles all market order processing including:
    1. Market-to-market matching
    2. Market-to-book matching
    3. Converting unfilled orders to aggressive limits
    """
    
    def __init__(self, *, order_book, agent_manager, order_state_manager, trade_execution_service,
                 logger, context, order_repository):
        # Pass context to parent class
        super().__init__(order_book=order_book, 
                         agent_manager=agent_manager, 
                         order_state_manager=order_state_manager,
                         context=context)
        
        self.context = context
        self.order_repository = order_repository
        self.trade_execution_service = trade_execution_service
        
        # Initialize services
        self._validation_service = OrderValidationService(
            commitment_calculator=agent_manager._commitment_calculator,
            current_price_provider=context,
            agent_manager=agent_manager,
            logger=logger,
            order_state_manager=order_state_manager
        )
        
        self._matching_service = OrderMatchingService(
            order_book=order_book,
            order_state_manager=order_state_manager,
            trade_execution_service=trade_execution_service,
            logger=logger,
            context=context
        )
        
        self._conversion_service = OrderConversionService(
            order_book=order_book,
            order_repository=order_repository,
            logger=logger
        )
        
        self._sync_service = AgentSyncService(
            agent_repository=agent_manager.agent_repository,
            order_repository=order_repository,
            logger=logger
        )
        
    def process_orders(self, market_orders: List[Order], current_price: float, 
                      round_number: int) -> Tuple[List[Trade], List[Order]]:
        """Main entry point for processing market orders"""
        if not market_orders:
            LoggingService.get_logger('market').info("No market orders to process")
            return [], []

        LoggingService.get_logger('market').info(f"\n=== Processing Market Orders (Round {round_number}) ===")
        
        # 1. Validation
        validated_orders = self._validate_orders(market_orders)
        if not validated_orders:
            return [], []
            
        # 2. Matching
        match_result = self._matching_service.match_market_orders(
            validated_orders, 
            current_price
        )
        
        # 3. Convert unfilled orders
        aggressive_limits = []
        if match_result.unfilled_orders:
            LoggingService.get_logger('market').info("\n3. Converting unfilled orders to aggressive limits...")
            aggressive_limits = self._convert_to_aggressive_limits(
                match_result.unfilled_orders, 
                current_price
            )
        
        return match_result.trades, aggressive_limits
        
    def _validate_orders(self, market_orders: List[Order]) -> List[Order]:
        """Validate market orders"""
        validated_orders = []
        for order in market_orders:
            validation_result = self._validation_service.validate_market_order(order)
            if validation_result.is_valid:
                self.order_state_manager.transition_to_matching(order)
                validated_orders.append(order)
            else:
                self.order_state_manager.handle_single_order_cancellation(
                    order=order, 
                    message=validation_result.message
                )

        return validated_orders
        
    def _convert_to_aggressive_limits(self, unfilled_orders: List[Order], 
                                    current_price: float) -> List[Order]:
        """Convert unfilled market orders to aggressive limit orders"""
        aggressive_limits = []
        
        for order in unfilled_orders:
            result = self._conversion_service.convert_to_aggressive_limit(order, current_price)
            
            if result.success and result.new_order:
                aggressive_limits.append(result.new_order)
                
                # Use sync service instead of direct sync
                sync_result = self._sync_service.sync_agent_orders_from_order_repository(order.agent_id)
                if not sync_result.success:
                    LoggingService.get_logger('market').warning(
                        f"Failed to sync orders for Agent {order.agent_id}: {sync_result.message}"
                    )
                
                LoggingService.get_logger('market').info(
                    f"Converting market order to aggressive limit - "
                    f"Transferred commitment: {result.transferred_commitment}, "
                    f"New price: {result.new_order.price:.2f}, "
                    f"Quantity: {result.new_order.quantity}"
                )
            else:
                LoggingService.get_logger('market').warning(
                    f"Failed to convert order {order.order_id}: {result.message}"
                )
                
        return aggressive_limits

        