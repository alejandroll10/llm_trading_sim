from dataclasses import dataclass
from typing import List, Tuple
from market.orders.order import Order, OrderState
from market.trade import Trade
import math

@dataclass
class MatchingResult:
    trades: List[Trade]
    remaining_orders: List[Order]
    unfilled_orders: List[Order]

class OrderMatchingService:
    def __init__(self, order_book, order_state_manager, trade_execution_service, logger, context):
        self._order_book = order_book
        self._order_state_manager = order_state_manager
        self.trade_execution_service = trade_execution_service
        self._logger = logger
        self.context = context

    def match_market_orders(self, orders: List[Order], current_price: float) -> MatchingResult:
        """Main matching orchestration"""
        # 1. Market-to-market matching
        market_trades, remaining_orders = self._net_market_orders(orders, current_price)
        
        # 2. Market-to-book matching
        book_trades, unfilled_orders = self._match_against_book(remaining_orders)
        
        return MatchingResult(
            trades=market_trades + book_trades,
            remaining_orders=remaining_orders,
            unfilled_orders=unfilled_orders
        )

    def _net_market_orders(self, market_orders: List[Order], current_price: float) -> Tuple[List[Trade], List[Order]]:
        """Match market buy and sell orders against each other"""
        trades = []
        remaining_orders = []
        
        # Split and sort orders
        buy_orders = sorted([o for o in market_orders if o.side == 'buy'], 
                           key=lambda x: x.timestamp)
        sell_orders = sorted([o for o in market_orders if o.side == 'sell'],
                            key=lambda x: x.timestamp)
        
        while buy_orders and sell_orders:
            buy = buy_orders[0]
            sell = sell_orders[0]
            
            trade_qty = min(buy.remaining_quantity, sell.remaining_quantity)
            trade_cost = trade_qty * current_price
            
            # Check commitment
            if not self._is_commitment_sufficient(buy, trade_cost):
                trade_qty = math.floor(buy.current_cash_commitment / current_price)
            
            if trade_qty <= 0:
                self._logger.info("Insufficient commitment for trade")
                self._logger.info("Skipping trade with zero quantity")
                if buy.remaining_quantity <= 0:
                    buy_orders.pop(0)
                if sell.remaining_quantity <= 0:
                    sell_orders.pop(0)
                continue
            
            trade = Trade.from_orders(buy_order=buy, sell_order=sell, quantity=trade_qty, price=current_price, round=self.context.round_number)
            trades.append(trade)
            
            self.trade_execution_service.handle_trade_execution(trade)
            
            if buy.remaining_quantity == 0:
                buy_orders.pop(0)
            if sell.remaining_quantity == 0:
                sell_orders.pop(0)
        
        remaining_orders.extend(buy_orders)
        remaining_orders.extend(sell_orders)
        
        return trades, remaining_orders

    def _match_against_book(self, market_orders: List[Order]) -> Tuple[List[Trade], List[Order]]:
        """Match remaining market orders against the order book"""
        trades = []
        unfilled_orders = []
        
        for order in market_orders:
            if order.side == 'buy':
                order_trades, remaining_qty = self._match_market_buy(order)
            else:
                order_trades, remaining_qty = self._match_market_sell(order)
                
            trades.extend(order_trades)
            
            self._handle_order_state_update(order, order_trades, remaining_qty, unfilled_orders)
                
        return trades, unfilled_orders

    def _handle_order_state_update(self, order: Order, trades: List[Trade], 
                                 remaining_qty: float, unfilled_orders: List[Order]):
        """Handle order state transitions after matching"""
        if not trades and remaining_qty == order.quantity:
            unfilled_orders.append(order)
        elif remaining_qty == 0:
            order.state = OrderState.FILLED
        elif remaining_qty < order.quantity:
            order.state = OrderState.PARTIALLY_FILLED
            order.remaining_quantity = remaining_qty
            unfilled_orders.append(order)
        else:
            order.remaining_quantity = remaining_qty
            unfilled_orders.append(order)

    def _is_commitment_sufficient(self, order: Order, trade_cost: float) -> bool:
        """Check if order has sufficient commitment"""
        if order.side == 'buy':
            return order.current_cash_commitment >= trade_cost
        return order.current_share_commitment >= trade_cost
            
    def _match_market_buy(self, order: Order) -> Tuple[List[Trade], float]:
        """Match a market buy order against the order book"""
        trades = []
        remaining_qty = order.remaining_quantity
        remaining_commitment = order.current_cash_commitment
        
        while remaining_qty > 0 and self._order_book.sell_orders:
            best_sell = self._order_book.pop_best_sell()
            sell_order = best_sell.order
            
            # Calculate trade quantity based on commitment
            affordable_qty = math.floor(remaining_commitment / sell_order.price)
            trade_qty = min(affordable_qty, remaining_qty, sell_order.remaining_quantity)
            
            if trade_qty <= 0:
                self._order_book.push_sell(best_sell)
                break
            
            # Create and record trade
            trade = Trade.from_orders(
                buy_order=order, 
                sell_order=sell_order, 
                quantity=trade_qty, 
                price=sell_order.price, 
                round=self.context.round_number
            )
            trades.append(trade)
            
            self.trade_execution_service.handle_trade_execution(trade)
            
            trade_cost = trade_qty * sell_order.price
            remaining_commitment -= trade_cost
            remaining_qty = order.remaining_quantity
            
            if sell_order.remaining_quantity > 0:
                self._order_book.push_sell(best_sell)
            
            if remaining_commitment <= 0:
                break
        
        return trades, remaining_qty
        
    def _match_market_sell(self, order: Order) -> Tuple[List[Trade], float]:
        """Match a market sell order against the order book"""
        trades = []
        remaining_qty = order.remaining_quantity
        
        while remaining_qty > 0 and self._order_book.buy_orders:
            best_buy = self._order_book.pop_best_buy()
            buy_order = best_buy.order
            
            trade_qty = min(remaining_qty, buy_order.remaining_quantity)
            
            trade = Trade.from_orders(
                buy_order=buy_order, 
                sell_order=order, 
                quantity=trade_qty, 
                price=abs(buy_order.price), 
                round=self.context.round_number
            )
            trades.append(trade)
            
            self.trade_execution_service.handle_trade_execution(trade)
            
            remaining_qty = order.remaining_quantity
            
            if buy_order.remaining_quantity > 0:
                self._order_book.push_buy(best_buy)
        
        return trades, remaining_qty
