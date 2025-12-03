from services.logging_service import LoggingService
from market.trade import Trade

class TradeProcessingService:
    def __init__(self, agent_manager, order_state_manager, order_book, order_repository, agent_repository, context, trade_execution_service=None):
        self.agent_manager = agent_manager
        self.order_state_manager = order_state_manager
        self.order_book = order_book
        self.order_repository = order_repository
        self.agent_repository = agent_repository
        self.context = context
        self.trade_execution_service = trade_execution_service

    def _would_cross_market(self, order) -> bool:
        """Check if a limit order would cross with existing orders"""
        if order.side == 'buy':
            best_ask = self.order_book.peek_best_sell()
            return best_ask is not None and order.price >= best_ask.price
        else:  # sell order
            best_bid = self.order_book.peek_best_buy()
            return best_bid is not None and order.price <= abs(best_bid.price)

    def _match_aggressive_limit(self, order) -> list:
        """Match an aggressive limit order against the book.

        Returns list of trades executed.
        """
        if not self.trade_execution_service:
            LoggingService.get_logger('market').error(
                "Cannot match aggressive limit: trade_execution_service is None"
            )
            return []

        trades = []

        if order.side == 'buy':
            # Match against sells
            while order.remaining_quantity > 0:
                best_sell = self.order_book.peek_best_sell()
                if not best_sell or order.price < best_sell.price:
                    break

                trade_qty = min(order.remaining_quantity, best_sell.remaining_quantity)
                if trade_qty <= 0:
                    break

                trade = Trade.from_orders(
                    buy_order=order,
                    sell_order=best_sell,
                    quantity=trade_qty,
                    price=best_sell.price,
                    round=self.context.round_number
                )
                trades.append(trade)

                # Execute trade (updates remaining_quantity on both orders)
                self.trade_execution_service.handle_trade_execution(trade)

                # Remove fully filled orders from book
                if best_sell.remaining_quantity == 0:
                    self.order_book.pop_best_sell()
        else:
            # Match against buys
            while order.remaining_quantity > 0:
                best_buy = self.order_book.peek_best_buy()
                if not best_buy or order.price > abs(best_buy.price):
                    break

                trade_qty = min(order.remaining_quantity, best_buy.remaining_quantity)
                if trade_qty <= 0:
                    break

                trade = Trade.from_orders(
                    buy_order=best_buy,
                    sell_order=order,
                    quantity=trade_qty,
                    price=abs(best_buy.price),
                    round=self.context.round_number
                )
                trades.append(trade)

                # Execute trade (updates remaining_quantity on both orders)
                self.trade_execution_service.handle_trade_execution(trade)

                # Remove fully filled orders from book
                if best_buy.remaining_quantity == 0:
                    self.order_book.pop_best_buy()

        return trades

    def process_aggressive_limits(self, new_aggressive_limits):
        """Handle processing of new aggressive limit orders.

        FIX for crossed market bug: Aggressive limits (created from unfilled market orders)
        can have prices that cross the book (e.g., buy at 110% of best ask). We must check
        for crossing and match against the book before adding.

        State flow:
        - If order would cross: COMMITTED → MATCHING → LIMIT_MATCHING → (match) → PENDING → ACTIVE
        - If order wouldn't cross: COMMITTED → MATCHING → LIMIT_MATCHING → PENDING → ACTIVE
        """
        all_trades = []

        for order in new_aggressive_limits:
            # Check if this order would cross the market BEFORE state transitions
            would_cross = self._would_cross_market(order)

            if would_cross:
                LoggingService.get_logger('market').info(
                    f"Aggressive limit {order.side} @ ${order.price:.2f} would cross market - matching first"
                )

                # Transition to matching states (allows PARTIALLY_FILLED transition)
                self.order_state_manager.transition_to_matching(
                    order,
                    notes="Starting aggressive limit matching"
                )
                self.order_state_manager.transition_to_limit_matching(
                    order,
                    notes="Ready for aggressive limit matching"
                )

                # Match against the book
                trades = self._match_aggressive_limit(order)
                all_trades.extend(trades)

                for trade in trades:
                    LoggingService.log_trade(trade, prefix="AggressiveLimit ")

                # If order still has remaining quantity, add to book
                if order.remaining_quantity > 0:
                    # Re-check if it would still cross after matching
                    if not self._would_cross_market(order):
                        # Transition to PENDING before adding to book
                        self.order_state_manager.transition_to_pending(
                            order,
                            notes="Moving to pending for book addition after aggressive limit matching"
                        )
                        self.order_book.add_limit_order(order)
                        self.order_state_manager.transition_to_active(order)
                    else:
                        # Still crosses - this shouldn't happen but log warning
                        LoggingService.get_logger('market').warning(
                            f"Aggressive limit still crosses after matching: "
                            f"{order.side} @ ${order.price:.2f}, remaining: {order.remaining_quantity}"
                        )
            else:
                # Order doesn't cross - use standard non-crossing flow
                self.order_state_manager.transition_non_crossing_limit(order)
                self.order_book.add_limit_order(order)
                self.order_state_manager.transition_to_active(order)

            self.order_state_manager.sync_agent_orders(order.agent_id)

        return new_aggressive_limits, all_trades

    def process_market_order_results(self, market_trades, new_aggressive_limits, limit_orders):
        """Process results from market order execution"""
        for trade in market_trades:
            LoggingService.log_trade(trade, prefix="Market ")

        processed_limits, aggressive_trades = self.process_aggressive_limits(new_aggressive_limits)
        limit_orders.extend(processed_limits)

        # Return all trades (market trades + aggressive limit matching trades)
        return market_trades + aggressive_trades

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
