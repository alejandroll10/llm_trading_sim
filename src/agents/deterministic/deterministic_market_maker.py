from typing import Dict, List
from agents.base_agent import BaseAgent
from agents.agents_api import TradeDecision, OrderType, OrderDetails

class DeterministicMarketMaker(BaseAgent):
    """Market maker that places both buy and sell orders based on fundamental value"""
    
    def __init__(self,
                 bid_discount: float = 0.01,  # 1% discount to market price for buys
                 ask_markup: float = 0.01,    # 1% markup to market price for sells
                 max_spread: float = 0.05,    # 5% maximum spread
                 fundamental_factor: float = 1.0,  # How much to adjust spread based on fundamental
                 buy_order_size: int = 100,   # Size of each buy order
                 sell_order_size: int = 100,  # Size of each sell order
                 position_limit_long: int = 1000000,  # Maximum long position
                 position_limit_short: int = 0,      # Maximum short position (negative)
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bid_discount = bid_discount
        self.ask_markup = ask_markup
        self.max_spread = max_spread
        self.fundamental_factor = fundamental_factor
        self.buy_order_size = buy_order_size
        self.sell_order_size = sell_order_size
        self.position_limit_long = position_limit_long
        self.position_limit_short = position_limit_short

    def calculate_prices(self, price: float, fundamental: float) -> tuple[float, float]:
        """Calculate bid and ask prices based on market price and fundamental value"""
        # Calculate percentage difference from fundamental
        fundamental_gap = (fundamental - price) / price
        
        # Adjust discount/markup based on gap to fundamental
        adjusted_discount = self.bid_discount - (fundamental_gap * self.fundamental_factor)
        adjusted_markup = self.ask_markup + (fundamental_gap * self.fundamental_factor)
        
        # Ensure discount/markup are within bounds
        final_discount = min(max(0.001, adjusted_discount), self.max_spread)
        final_markup = min(max(0.001, adjusted_markup), self.max_spread)
        
        bid_price = price * (1 - final_discount)
        ask_price = price * (1 + final_markup)
        
        return bid_price, ask_price

    def make_decision(self, market_state: Dict, history: List, round_number: int) -> TradeDecision:
        price = market_state['price']
        fundamental = market_state.get('fundamental_price', price)
        current_position = self.shares
        
        # Calculate bid and ask prices
        bid_price, ask_price = self.calculate_prices(price, fundamental)
        
        orders = []
        
        # Determine if we can place buy orders (below position limit)
        if current_position < self.position_limit_long:
            # Check available cash
            available_cash = self.available_cash
            max_shares = int(available_cash / bid_price)
            
            if max_shares > 0:
                # Adjust order size based on remaining position capacity
                buy_size = min(
                    self.buy_order_size,
                    self.position_limit_long - current_position,
                    max_shares
                )
                
                if buy_size > 0:
                    # Add buy order
                    orders.append(OrderDetails(
                        decision="Buy",
                        quantity=buy_size,
                        order_type=OrderType.LIMIT,
                        price_limit=bid_price
                    ))
        
        # Determine if we can place sell orders (above position limit)
        if current_position > self.position_limit_short:
            # Check available shares
            available_shares = self.available_shares
            
            if available_shares > 0:
                # Adjust order size based on remaining position capacity
                sell_size = min(
                    self.sell_order_size,
                    current_position - self.position_limit_short,
                    available_shares
                )
                
                if sell_size > 0:
                    # Add sell order
                    orders.append(OrderDetails(
                        decision="Sell",
                        quantity=sell_size,
                        order_type=OrderType.LIMIT,
                        price_limit=ask_price
                    ))
        
        # Create reasoning string with details of the orders
        if len(orders) == 0:
            reasoning = "No orders placed - at position limits or insufficient funds/shares"
        elif len(orders) == 1:
            order_type = orders[0].decision
            reasoning = f"Making market: {'Bid' if order_type == 'Buy' else 'Ask'} ${orders[0].price_limit:.2f}"
        else:
            reasoning = f"Making market: Bid ${bid_price:.2f}, Ask ${ask_price:.2f} (fundamental: ${fundamental:.2f})"
        
        return TradeDecision(
            orders=orders,
            replace_decision="Replace",
            reasoning=reasoning,
            valuation=fundamental,
            valuation_reasoning="Based on provided fundamental value",
            price_target=fundamental,
            price_target_reasoning="Target is aligned with fundamental value"
        ) 