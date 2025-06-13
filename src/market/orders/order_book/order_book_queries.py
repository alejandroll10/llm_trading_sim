from typing import Optional, Dict, List
from market.orders.order import Order

class OrderBookQueries:
    # No __init__ needed, it will use OrderBook's __init__

    def get_best_ask(self) -> Optional[float]:
        """Get the best (lowest) ask price"""
        if not self.sell_orders:
            return None
        return self.sell_orders[0].display_price

    def get_best_bid(self) -> Optional[float]:
        """Get the best (highest) bid price"""
        if not self.buy_orders:
            return None
        return self.buy_orders[0].display_price


    def peek_best_buy(self) -> Optional[Order]:
        """Get the best buy order without removing it"""
        if not self.buy_orders:
            return None
        return self.buy_orders[0].order

    def peek_best_sell(self) -> Optional[Order]:
        """Get the best sell order without removing it"""
        if not self.sell_orders:
            return None
        return self.sell_orders[0].order


    def calculate_price_from_book(self) -> float:
        """Calculate price from order book when no trades occur"""
        if self.sell_orders and not self.buy_orders:
            return self.get_best_ask()
        elif self.buy_orders and not self.sell_orders:
            return self.get_best_bid()
        elif self.buy_orders and self.sell_orders:
            return self.get_best_ask()
        return 0.0

    def get_aggregated_levels(self) -> Dict[str, List[Dict[str, float]]]:
        """Get aggregated buy and sell levels"""
        buy_levels = {}
        sell_levels = {}
        
        for entry in self.buy_orders:
            price = entry.display_price
            buy_levels[price] = buy_levels.get(price, 0) + entry.quantity
        
        for entry in self.sell_orders:
            price = entry.display_price
            sell_levels[price] = sell_levels.get(price, 0) + entry.quantity
        
        return {
            'buy_levels': [{'price': p, 'quantity': q} 
                          for p, q in sorted(buy_levels.items(), reverse=True)],
            'sell_levels': [{'price': p, 'quantity': q} 
                          for p, q in sorted(sell_levels.items())]
        }

    def estimate_market_order_cost(self, quantity: int, side: str) -> tuple[float, int]:
        """
        Estimate the total cost/proceeds of a market order by walking the book
        Returns: (total_cost, fillable_quantity)
        """
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
            
        # Handle empty book case
        if self.is_empty_for_side(side):
            reference_price = self.context.current_price
            return (quantity * reference_price if side == 'buy' 
                   else -quantity * reference_price), quantity
        
        total_cost = 0
        remaining_qty = quantity
        fillable_qty = 0
        
        # When buying, walk the sell orders (ascending price)
        # When selling, walk the buy orders (descending price)
        orders_to_walk = (sorted(self.sell_orders) if side == 'buy' 
                         else sorted(self.buy_orders, reverse=True))
        
        for entry in orders_to_walk:
            if remaining_qty <= 0:
                break
            
            fill_qty = min(remaining_qty, entry.order.remaining_quantity)
            
            # When buying, we pay the ask price
            # When selling, we receive the bid price
            price = entry.display_price
            total_cost += fill_qty * price if side == 'buy' else -fill_qty * price
            
            fillable_qty += fill_qty
            remaining_qty -= fill_qty
                
        return total_cost, fillable_qty

    def get_midpoint(self) -> Optional[float]:
        """Get the midpoint price, handling empty books"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid + best_ask) / 2.0

