from dataclasses import dataclass
from typing import List, Optional, Tuple
from market.orders.order import Order
from market.trade import Trade

@dataclass
class LimitMatchingResult:
    trades: List[Trade]
    remaining_quantity: int
    is_fully_matched: bool
    error: Optional[str] = None

class LimitMatchingService:
    def __init__(self, order_book, order_state_manager, trade_execution_service, logger, context):
        self._order_book = order_book
        self.order_state_manager = order_state_manager
        self.trade_execution_service = trade_execution_service
        self._logger = logger
        self._context = context
        
    def match_limit_order(self, order: Order) -> LimitMatchingResult:
        """Match a limit order against the book"""
        try:
            trades = []
            if order.side == 'buy':
                trades = self._match_limit_buy(order)
            else:
                trades = self._match_limit_sell(order)
                
            return LimitMatchingResult(
                trades=trades,
                remaining_quantity=order.remaining_quantity,
                is_fully_matched=order.remaining_quantity == 0
            )
        except Exception as e:
            self._logger.error(f"Error matching limit order: {str(e)}, order: {order} with history {order.print_history()}")
            
            return LimitMatchingResult(
                trades=[],
                remaining_quantity=order.remaining_quantity,
                is_fully_matched=False,
                error=str(e)
            )
        
    def _match_limit_buy(self, order: Order) -> List[Trade]:
        """Match a limit buy order against the order book"""
        trades = []
        while order.remaining_quantity > 0:
            best_sell = self._order_book.peek_best_sell()
            if not best_sell or order.price < best_sell.price:
                break
                
            trade_qty = min(order.remaining_quantity, best_sell.remaining_quantity)
            
            # Validate trade quantity
            self._validate_trade_quantity(
                trade_qty, 
                order, 
                best_sell,
                "buy"
            )
            
            # Create and execute trade
            trade = Trade.from_orders(
                buy_order=order, 
                sell_order=best_sell, 
                quantity=trade_qty, 
                price=best_sell.price,
                round=self._context.round_number
            )
            trades.append(trade)
            
            # Log pre-execution state
            pre_buy_qty = order.remaining_quantity
            pre_sell_qty = best_sell.remaining_quantity
            
            # Execute trade
            self.trade_execution_service.handle_trade_execution(trade)
            
            # Validate post-trade state
            self._validate_post_trade_state(
                order,
                best_sell,
                pre_buy_qty,
                pre_sell_qty,
                trade_qty
            )
            
            # Remove filled orders from book
            if best_sell.remaining_quantity == 0:
                self._order_book.pop_best_sell()
                
        return trades

    def _match_limit_sell(self, order: Order) -> List[Trade]:
        """Match a limit sell order against the order book"""
        trades = []
        while order.remaining_quantity > 0:
            best_buy = self._order_book.peek_best_buy()
            if not best_buy or order.price > abs(best_buy.price):
                break
                
            trade_qty = min(order.remaining_quantity, best_buy.remaining_quantity)
            
            # Validate trade quantity
            self._validate_trade_quantity(
                trade_qty, 
                order, 
                best_buy,
                "sell"
            )
            
            # Create and execute trade
            trade = Trade.from_orders(
                buy_order=best_buy, 
                sell_order=order, 
                quantity=trade_qty, 
                price=abs(best_buy.price), 
                round=self._context.round_number
            )
            trades.append(trade)
            
            # Log pre-execution state
            pre_sell_qty = order.remaining_quantity
            pre_buy_qty = best_buy.remaining_quantity
            
            # Execute trade
            self.trade_execution_service.handle_trade_execution(trade)
            
            # Validate post-trade state
            self._validate_post_trade_state(
                order,
                best_buy,
                pre_sell_qty,
                pre_buy_qty,
                trade_qty
            )
            
            # Remove filled orders from book
            if best_buy.remaining_quantity == 0:
                self._order_book.pop_best_buy()
                
        return trades
        
    def _validate_trade_quantity(self, trade_qty: int, order1: Order, 
                               order2: Order, side: str) -> None:
        """Validate trade quantity before execution"""
        if trade_qty <= 0:
            error_msg = (
                f"Invalid trade quantity calculated:"
                f"\n{side.capitalize()} order remaining: {order1.remaining_quantity}"
                f"\nCounter order remaining: {order2.remaining_quantity}"
                f"\nCalculated trade_qty: {trade_qty}"
                f"\n\nOrder 1 History:\n{order1.print_history()}"
                f"\n\nOrder 2 History:\n{order2.print_history()}"
            )
            self._logger.error(error_msg)
            raise ValueError(error_msg)
            
    def _validate_post_trade_state(self, order1: Order, order2: Order,
                                 pre_qty1: int, pre_qty2: int, 
                                 trade_qty: int) -> None:
        """Validate order states after trade execution"""
        if order1.remaining_quantity < 0 or order2.remaining_quantity < 0:
            error_msg = (
                f"Negative remaining quantity after trade:"
                f"\nOrder 1 remaining: {order1.remaining_quantity}"
                f"\nOrder 2 remaining: {order2.remaining_quantity}"
                f"\nTrade quantity was: {trade_qty}"
                f"\nPre-trade quantities: {pre_qty1}, {pre_qty2}"
                f"\n\nOrder 1 History:\n{order1.print_history()}"
                f"\n\nOrder 2 History:\n{order2.print_history()}"
            )
            self._logger.error(error_msg)
            raise ValueError(error_msg)