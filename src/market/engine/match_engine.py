import random
from market.orders.handlers.market_handler import MarketOrderHandler
from market.orders.handlers.limit_handler import LimitOrderHandler
from market.orders.order import Order
from typing import List
from market.engine.market_result import MarketResult
from market.engine.services.order_processing_service import OrderProcessingService
from market.engine.services.trade_processing_service import TradeProcessingService
from services.logging_service import LoggingService

class MatchingEngine:
    def __init__(self, order_book, agent_manager, agent_repository, order_repository, context, order_state_manager = None, logger=None, trades_logger=None, trade_execution_service=None, is_multi_stock=False):
        self.order_book = order_book
        self.agent_manager = agent_manager
        self.order_repository = order_repository
        self.order_state_manager = order_state_manager
        self.agent_repository = agent_repository
        self.trade_execution_service = trade_execution_service
        self.context = context
        self.is_multi_stock = is_multi_stock  # Flag to indicate multi-stock mode
        
        # Initialize services
        self.order_processing_service = OrderProcessingService(order_book)
        self.trade_processing_service = TradeProcessingService(
            agent_manager, 
            order_state_manager, 
            order_book, 
            order_repository, 
            agent_repository, 
            context
        )
        
        # Initialize handlers with services
        self.market_order_handler = MarketOrderHandler(
            order_book=order_book,
            agent_manager=agent_manager,
            trade_execution_service=self.trade_execution_service,
            logger=logger,
            context=self.context,
            order_repository=order_repository,
            order_state_manager=order_state_manager
        )
        
        self.limit_order_handler = LimitOrderHandler(
            order_book=order_book,
            agent_manager=agent_manager,
            trade_execution_service=self.trade_execution_service,
            logger=logger,
            order_repository=order_repository,
            order_state_manager=order_state_manager,
            context=self.context
        )

    def match_orders(self, new_orders: List[Order], current_price: float, round_number: int) -> MarketResult:
        """Match orders for a trading round with prioritized processing"""
        # Log pre-match state
        LoggingService.log_market_state(self.order_book, round_number, "Pre-Match State")

        # Separate liquidation orders initiated by broker
        liquidation_orders = [o for o in new_orders if getattr(o, 'liquidation', False)]
        regular_orders = [o for o in new_orders if not getattr(o, 'liquidation', False)]

        trades = []

        # Process liquidation orders with highest priority
        if liquidation_orders:
            liq_trades, _ = self.market_order_handler.process_orders(
                liquidation_orders, current_price, round_number
            )
            trades.extend(
                self.trade_processing_service.process_market_order_results(
                    liq_trades, [], []
                )
            )
            LoggingService.log_market_state(
                self.order_book, round_number, "Post-Liquidation Order State"
            )

        # Split remaining orders by type
        market_orders, limit_orders = self.order_processing_service.split_orders_by_type(regular_orders)
        
        # Handle limit orders
        crossing_orders = self.limit_order_handler.add_non_crossing_orders(limit_orders)
        LoggingService.log_market_state(self.order_book, round_number, "Post-Limit Order State")
        
        # Process market orders
        if market_orders:
            market_trades, new_aggressive_limits = self.market_order_handler.process_orders(
                market_orders, current_price, round_number
            )
            trades.extend(self.trade_processing_service.process_market_order_results(
                market_trades, new_aggressive_limits, limit_orders
            ))
            LoggingService.log_market_state(self.order_book, round_number, "Post-Market Order State")
        
        # Process crossing limit orders
        if crossing_orders:
            limit_trades = self.limit_order_handler.process_orders(crossing_orders)
            trades.extend(self.trade_processing_service.process_limit_order_results(limit_trades))
        
        # Log trades and calculate new price
        LoggingService.log_trades(trades, round_number)
        new_price = self.trade_processing_service.calculate_new_price(trades, current_price)
        
        # Update agent wealth and log final state
        # In multi-stock mode, wealth is updated centrally with all stock prices
        if not self.is_multi_stock:
            self.agent_repository.update_all_wealth(new_price)
        LoggingService.log_market_state(self.order_book, round_number, "End of Round State")
        
        return MarketResult(
            trades=trades,
            price=new_price,
            volume=sum(t.quantity for t in trades)
        )