"""Market calculation logic for computing derived values"""
from typing import List, Dict, Tuple, Any
from market.information.information_types import InformationSignal
from market.trade import Trade


class MarketCalculator:
    """Performs calculations and aggregations on market data"""

    @staticmethod
    def calculate_pf_ratio(price_signal: InformationSignal,
                          fundamental_signal: InformationSignal) -> float:
        """Calculate price/fundamental ratio

        Args:
            price_signal: Current price signal
            fundamental_signal: Fundamental value signal

        Returns:
            Price/fundamental ratio, or None if calculation not possible
        """
        try:
            if fundamental_signal.value is None or fundamental_signal.value == 0:
                return None
            return price_signal.value / fundamental_signal.value
        except Exception:
            return None

    @staticmethod
    def consolidate_order_levels(levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate orders at same price level

        Args:
            levels: List of order levels with 'price' and 'quantity'

        Returns:
            Consolidated and sorted list of price levels
        """
        price_map = {}
        for level in levels:
            # Normalize price to avoid floating-point precision issues
            price = round(float(level['price']), 2)  # Round to 2 decimal places
            if price in price_map:
                price_map[price] += int(level['quantity'])
            else:
                price_map[price] = int(level['quantity'])

        # Return sorted levels
        return sorted(
            [{'price': price, 'quantity': quantity} for price, quantity in price_map.items()],
            key=lambda x: x['price']
        )

    @staticmethod
    def aggregate_and_sort_orders(order_list: List[Dict[str, Any]],
                                  is_buy: bool) -> Tuple[List[Dict[str, Any]], int]:
        """Aggregate and sort orders by price

        Args:
            order_list: List of orders with 'price' and 'quantity'
            is_buy: True if buy orders (sort high to low), False if sell (low to high)

        Returns:
            Tuple of (aggregated_orders, market_order_quantity)
        """
        price_map = {}
        market_orders_quantity = 0

        for order in order_list:
            price = order.get('price')
            quantity = int(order['quantity'])
            if price is not None:
                # Normalize price to avoid floating-point precision issues
                price = round(float(price), 2)
                if price in price_map:
                    price_map[price] += quantity
                else:
                    price_map[price] = quantity
            else:
                # Market orders (no price specified)
                market_orders_quantity += quantity

        # Convert to list of dicts
        orders_with_price = [{'price': price, 'quantity': q} for price, q in price_map.items()]

        # Sort orders: for buys, from highest to lowest price; for sells, lowest to highest
        orders_with_price.sort(key=lambda x: x['price'], reverse=is_buy)

        return orders_with_price, market_orders_quantity

    @staticmethod
    def calculate_trade_pnl_summary(trade_history: List[Trade],
                                   agent_id: str,
                                   current_round: int,
                                   lookback: int = 5) -> Dict[str, Any]:
        """Calculate P&L summary from trade history

        Args:
            trade_history: List of trades
            agent_id: ID of the agent
            current_round: Current round number
            lookback: Number of rounds to look back

        Returns:
            Dictionary containing trade summary with buy/sell volumes and average prices
        """
        if current_round == 0 or not trade_history:
            return {
                'recent_trades': [],
                'buy_volume': 0,
                'buy_value': 0,
                'sell_volume': 0,
                'sell_value': 0,
                'avg_buy_price': 0,
                'avg_sell_price': 0
            }

        # Get trades from recent rounds
        recent_trades = sorted(
            [t for t in trade_history if t.round < current_round],
            key=lambda x: x.round,
            reverse=True
        )[:lookback]

        # Calculate running totals for P&L tracking
        buy_volume = 0
        buy_value = 0
        sell_volume = 0
        sell_value = 0
        trade_details = []

        for trade in recent_trades:
            # Determine if agent was buyer or seller
            is_buyer = trade.buyer_id == agent_id
            role = "Bought" if is_buyer else "Sold"

            # Update running totals
            if is_buyer:
                buy_volume += trade.quantity
                buy_value += trade.value
            else:
                sell_volume += trade.quantity
                sell_value += trade.value

            trade_details.append({
                'round': trade.round,
                'role': role,
                'quantity': trade.quantity,
                'price': trade.price,
                'value': trade.value
            })

        # Calculate average prices
        avg_buy_price = buy_value / buy_volume if buy_volume > 0 else 0
        avg_sell_price = sell_value / sell_volume if sell_volume > 0 else 0

        return {
            'recent_trades': trade_details,
            'buy_volume': buy_volume,
            'buy_value': buy_value,
            'sell_volume': sell_volume,
            'sell_value': sell_value,
            'avg_buy_price': avg_buy_price,
            'avg_sell_price': avg_sell_price
        }
