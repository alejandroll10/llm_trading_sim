from services.logging_service import LoggingService

class TradeProcessingService:
    def __init__(self, agent_manager, order_state_manager, order_book, order_repository, agent_repository, context):
        self.agent_manager = agent_manager
        self.order_state_manager = order_state_manager
        self.order_book = order_book
        self.order_repository = order_repository
        self.agent_repository = agent_repository
        self.context = context

    def process_aggressive_limits(self, new_aggressive_limits):
        """Handle processing of new aggressive limit orders"""
        for order in new_aggressive_limits:
            self.order_state_manager.transition_non_crossing_limit(order)
            self.order_book.add_limit_order(order)
            self.order_state_manager.transition_to_active(order)
            self.order_state_manager.sync_agent_orders(order.agent_id)
        return new_aggressive_limits

    def process_market_order_results(self, market_trades, new_aggressive_limits, limit_orders):
        """Process results from market order execution"""
        for trade in market_trades:
            LoggingService.log_trade(trade, prefix="Market ")
            
        processed_limits = self.process_aggressive_limits(new_aggressive_limits)
        limit_orders.extend(processed_limits)
        
        return market_trades

    def process_limit_order_results(self, limit_trades):
        """Process results from limit order execution"""
        for trade in limit_trades:
            LoggingService.log_trade(trade, prefix="Limit ")
        return limit_trades
    
    def calculate_new_price(self, trades, current_price: float) -> float:
        """Calculate the new price based on executed trades or market conditions."""
        if trades:
            # Add all new trades to history
            for trade in trades:
                self.context.add_trade(trade)
            
            # Could implement different price strategies
            return self._calculate_price_from_trades(trades)
        
        # If no new trades, use order book for price discovery
        book_price = self.order_book.calculate_price_from_book()
        if book_price > 0:
            return book_price
            
        # If no trades and no valid book price, maintain current price
        return current_price

    def _calculate_price_from_trades(self, trades, strategy: str = 'trade') -> float:
        """Calculate price using configurable strategies"""
        
        if strategy == 'vwap_current':
            return self._calculate_vwap(trades)
        elif strategy == 'vwap_window':
            # Get recent trades including history
            recent_trades = self._get_trades_in_window(
                window_size=self.config.get('vwap_window_size', 5)
            )
            return self._calculate_vwap(recent_trades)
        else:  # default to last trade
            return trades[-1].price

    def _get_trades_in_window(self, window_size: int) -> list:
        """Get trades from recent rounds"""
        history = self.context.public_info['trade_history']
        current_round = self.context.round_number
        
        # Filter trades within window
        recent_trades = [
            trade for trade in history 
            if trade['round'] >= current_round - window_size
        ]
        return recent_trades

    def _calculate_vwap(self, trades) -> float:
        """Calculate volume-weighted average price"""
        if not trades:
            return 0.0
            
        total_value = sum(trade['price'] * trade['quantity'] for trade in trades)
        total_volume = sum(trade['quantity'] for trade in trades)
        
        return total_value / total_volume if total_volume > 0 else 0.0
