from typing import Dict, Any, List
from dataclasses import dataclass
from market.information.information_types import InformationType, InformationSignal
from market.trade import Trade
from agents.LLMs.llm_prompt_templates import (
    TRADING_OPTIONS_TEMPLATE, BASE_MARKET_TEMPLATE, 
    POSITION_INFO_TEMPLATE, DIVIDEND_INFO_TEMPLATE, 
    INTEREST_INFO_TEMPLATE, PRICE_HISTORY_TEMPLATE,
    REDEMPTION_INFO_TEMPLATE
)

@dataclass
class AgentContext:
    """Agent's current state and context"""
    agent_id: str
    cash: float
    shares: int
    available_cash: float
    available_shares: int
    outstanding_orders: Dict[str, List[Dict[str, Any]]]
    signal_history: Dict[int, Dict[InformationType, InformationSignal]] = None  # Make it optional with default None
    trade_history: List[Trade] = None  # Add trade history
    dividend_cash: float = 0.0
    committed_cash: float = 0.0
    committed_shares: int = 0

class MarketStateFormatter:
    """Formats market state information for LLM consumption"""
    
    # Constants for formatting
    CURRENCY_FORMAT = "${:.2f}"
    PERCENT_FORMAT = "{:.2f}%"
    NO_HISTORY = "No History"
    UNAVAILABLE = "Unavailable"

    @staticmethod
    def format_prompt_sections(
        agent_signals: Dict[InformationType, InformationSignal],
        agent_context: AgentContext,
        signal_history: Dict[int, Dict[InformationType, InformationSignal]] = None
    ) -> Dict[str, str]:
        """Format all sections of the LLM prompt using information signals"""
        try:
            # Extract basic signals
            price_signal = agent_signals[InformationType.PRICE]
            volume_signal = agent_signals[InformationType.VOLUME]
            order_book_signal = agent_signals[InformationType.ORDER_BOOK]
            fundamental_signal = agent_signals[InformationType.FUNDAMENTAL]
            dividend_signal = agent_signals[InformationType.DIVIDEND]
            interest_signal = agent_signals[InformationType.INTEREST]

            periods_remaining = fundamental_signal.metadata['periods_remaining']
            if isinstance(periods_remaining, str):
                if periods_remaining.isdigit():
                    periods_remaining = int(periods_remaining)
            if isinstance(periods_remaining, int) and periods_remaining > 0:
                num_rounds = periods_remaining + price_signal.metadata['round']
            else:
                num_rounds = "Infinite"
            
            context = {
                # Market data from signals
                'price': price_signal.value,
                'round_number': price_signal.metadata['round'],
                'num_rounds': num_rounds,
                
                # Agent data (already safe via dataclass)
                'shares': agent_context.shares,
                'cash': agent_context.cash,
                'dividend_cash': agent_context.dividend_cash,
                'total_available_cash': agent_context.available_cash,
                'committed_cash': agent_context.committed_cash,
                'committed_shares': agent_context.committed_shares,
                
                # Calculated display values
                'volume_display': MarketStateFormatter._format_volume(volume_signal),
                'fundamental_display': MarketStateFormatter._format_fundamental(fundamental_signal),
                'order_book_display': MarketStateFormatter._format_order_book(order_book_signal),
                'orders_display': MarketStateFormatter._format_outstanding_orders(
                    agent_context.outstanding_orders
                ),
                'pf_ratio_display': MarketStateFormatter._format_pf_ratio(
                    price_signal, fundamental_signal
                ),
                
                # Dividend and interest data
                **MarketStateFormatter._prepare_dividend_context(dividend_signal),
                **MarketStateFormatter._prepare_interest_context(interest_signal),
                
                # Add redemption context
                **MarketStateFormatter._prepare_redemption_context(fundamental_signal),
                
                # Add final round message
                'final_round_message': "\nIn the final round, all shares are redeemed at the fundamental value." if num_rounds != "Infinite" else ""
            }
            
            # Use signal_history from context if not provided directly
            history_to_use = signal_history or agent_context.signal_history or {}
            
            # Format price history
            price_history = MarketStateFormatter._format_price_history(
                current_round=price_signal.metadata['round'],
                signal_history=history_to_use
            )
            
            # Add to context with the correct key name
            context['price_history'] = price_history
            
            # Add trade history
            context['trade_history'] = MarketStateFormatter._format_trade_history(
                trade_history=agent_context.trade_history,
                agent_id=agent_context.agent_id,
                current_round=price_signal.metadata['round'],
                lookback=5
            )
            
            # Use templates consistently
            return {
                'base_market_state': BASE_MARKET_TEMPLATE.format(**context),
                'position_info': POSITION_INFO_TEMPLATE.format(**context),
                'price_history': PRICE_HISTORY_TEMPLATE.format(**context),
                'dividend_info': DIVIDEND_INFO_TEMPLATE.format(**context),
                'interest_info': INTEREST_INFO_TEMPLATE.format(**context),
                'redemption_info': REDEMPTION_INFO_TEMPLATE.format(**context),
                'trading_options': TRADING_OPTIONS_TEMPLATE
            }
            
        except KeyError as e:
            raise ValueError(f"Missing required signal: {e}")

    @staticmethod
    def _format_currency(value: float) -> str:
        """Consistent currency formatting"""
        return MarketStateFormatter.CURRENCY_FORMAT.format(value)

    @staticmethod
    def _format_percent(value: float) -> str:
        """Consistent percentage formatting"""
        return MarketStateFormatter.PERCENT_FORMAT.format(value)

    @staticmethod
    def _format_volume(volume_signal: InformationSignal) -> str:
        """Format volume display from signal"""
        if volume_signal.metadata['round'] == 0:
            return MarketStateFormatter.NO_HISTORY
        return f"{volume_signal.value:.2f}"

    @staticmethod
    def _format_pf_ratio(price_signal: InformationSignal, 
                        fundamental_signal: InformationSignal) -> str:
        """Format price/fundamental ratio"""
        try:
            if fundamental_signal.value is None or fundamental_signal.value == 0:
                return MarketStateFormatter.UNAVAILABLE
            ratio = price_signal.value / fundamental_signal.value
            return MarketStateFormatter._format_currency(ratio)
        except Exception:
            return MarketStateFormatter.UNAVAILABLE

    @staticmethod
    def _format_order_book(order_book_signal: InformationSignal) -> str:
        """Format order book from signal"""
        lines = []
        order_book = order_book_signal.value
        metadata = order_book_signal.metadata
        
        # Add best bid/ask if available
        if metadata.get('best_bid'):
            lines.append(f"Best Bid: {MarketStateFormatter._format_currency(metadata['best_bid'])}")
        if metadata.get('best_ask'):
            lines.append(f"Best Ask: {MarketStateFormatter._format_currency(metadata['best_ask'])}")
        
        # Helper function to consolidate orders at same price
        def consolidate_levels(levels):
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
        
        # Format sell side (highest to lowest price)
        sell_levels = order_book.get('sell_levels', [])
        if sell_levels:
            lines.append("\nSell Orders:")
            consolidated_sells = consolidate_levels(sell_levels)
            # Sell orders should be from highest to lowest price
            for level in reversed(consolidated_sells):
                lines.append(
                    f"- {level['quantity']} shares @ {MarketStateFormatter._format_currency(level['price'])}"
                )
        else:
            lines.append("\nNo sell orders")
                
        # Format buy side (highest to lowest price)
        buy_levels = order_book.get('buy_levels', [])
        if buy_levels:
            lines.append("\nBuy Orders:")
            consolidated_buys = consolidate_levels(buy_levels)
            # Buy orders should also be from highest to lowest price
            for level in reversed(consolidated_buys):
                lines.append(
                    f"- {level['quantity']} shares @ {MarketStateFormatter._format_currency(level['price'])}"
                )
        else:
            lines.append("\nNo buy orders")
                
        return "\n".join(lines)

    @staticmethod
    def _format_fundamental(fundamental_signal: InformationSignal) -> str:
        """Format fundamental price from signal"""
        if fundamental_signal.value is None:
            return "Unavailable"
        return f"${fundamental_signal.value:.2f}"

    @staticmethod
    def _prepare_dividend_context(dividend_signal: InformationSignal) -> Dict[str, Any]:
        """Prepare dividend context from signal"""
        metadata = dividend_signal.metadata
        yields = metadata['yields']
        model = metadata.get('model', {})
        
        return {
            # Current status
            'expected_dividend': dividend_signal.value,
            'expected_yield': yields['expected'],
            'last_paid_text': (
                f"${metadata['last_paid_dividend']:.2f}"
                if metadata.get('last_paid_dividend') is not None 
                else "No dividends paid yet"
            ),
            
            # Dividend model details
            'base_dividend': dividend_signal.value,  # Use expected as base if not provided
            'variation': metadata.get('variation', 0.0),
            'max_dividend': metadata.get('max_dividend', 0.0),
            'min_dividend': metadata.get('min_dividend', 0.0),
            'max_yield': yields['max'],
            'min_yield': yields['min'],
            'probability_percent': metadata.get('probability', 50.0),
            'inverse_probability_percent': metadata.get('probability', 50.0),
            
            # Payment info
            'next_payment_round': metadata['next_payment_round'],
            'should_pay': metadata['should_pay'],
            'dividend_destination': metadata.get('destination', 'dividend'),
            'dividend_tradeable': metadata.get('tradeable', 'non-tradeable')
        }

    @staticmethod
    def _prepare_interest_context(interest_signal: InformationSignal) -> Dict[str, Any]:
        """Prepare interest context from signal"""
        metadata = interest_signal.metadata
        interest_destination = metadata.get('interest_destination', 'main')
        
        return {
            'interest_rate': interest_signal.value * 100,  # Convert to percentage
            'compound_frequency': metadata['compound_frequency'],
            'last_payment': metadata.get('last_payment'),
            'next_payment_round': metadata.get('next_payment_round'),
            'interest_destination': interest_destination,
            'interest_tradeable': "available for trading" if interest_destination == 'main' else "separate from trading"
        }

    @staticmethod
    def _format_outstanding_orders(orders: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format agent's outstanding orders with aggregation and ordering"""
        output = "Your Outstanding Orders:"
        
        if not orders['buy'] and not orders['sell']:
            return output + "\nNo outstanding orders"
        
        # Helper function to aggregate and sort orders
        def aggregate_and_sort_orders(order_list, is_buy):
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

        if orders['sell']:
            output += "\nSell Orders:"
            aggregated_sells, market_sells = aggregate_and_sort_orders(orders['sell'], is_buy=False)
            if market_sells > 0:
                output += f"\n- {market_sells} shares (market order)"
            for order in aggregated_sells:
                output += f"\n- {order['quantity']} shares @ {MarketStateFormatter._format_currency(order['price'])}"

        if orders['buy']:
            output += "\nBuy Orders:"
            aggregated_buys, market_buys = aggregate_and_sort_orders(orders['buy'], is_buy=True)
            if market_buys > 0:
                output += f"\n- {market_buys} shares (market order)"
            for order in aggregated_buys:
                output += f"\n- {order['quantity']} shares @ {MarketStateFormatter._format_currency(order['price'])}"
                    
        return output

    @staticmethod
    def _format_percent(value: float) -> str:
        return MarketStateFormatter.PERCENT_FORMAT.format(value)

    @staticmethod
    def _format_price_history(
        current_round: int,
        signal_history: Dict[int, Dict[InformationType, InformationSignal]],
        lookback: int = 5
    ) -> str:
        """Format price history for display"""
        if current_round <= 1:  # First round
            return "No price history available (first round)"
            
        if not signal_history:
            return "No price history available"
            
        # Get rounds to display
        rounds = sorted(
            [r for r in signal_history.keys() if r < current_round],
            reverse=True
        )[:lookback]
        
        if not rounds:
            return "No previous price data"
            
        # Format each round's price with volume if available
        history_lines = []
        for round_num in rounds:
            price_info = signal_history[round_num].get(InformationType.PRICE)
            volume_info = signal_history[round_num].get(InformationType.VOLUME)
            
            if price_info:
                line = f"Round {round_num}: {MarketStateFormatter._format_currency(price_info.value)}"
                if volume_info:
                    line += f" (Volume: {volume_info.value:.0f})"
                history_lines.append(line)
                    
        return "\n".join(history_lines)

    @staticmethod
    def _format_trade_history(
        trade_history: List[Trade],
        agent_id: str,
        current_round: int,
        lookback: int = 5
    ) -> str:
        """Format recent trade history for display with P&L information
        
        Args:
            trade_history: List of trades to format
            agent_id: ID of the agent requesting history
            current_round: Current round number
            lookback: Number of rounds to look back
        """

        if current_round == 0:  # First round is 0 (Python is 0 indexed)
            return "No trade history available (first round)"
        
        if not trade_history:
            return "No trade history available"
        
        # Get trades from recent rounds
        recent_trades = sorted(
            [t for t in trade_history if t.round < current_round],
            key=lambda x: x.round,
            reverse=True
        )[:lookback]
        
        if not recent_trades:
            return "No recent trade data"
        
        # Calculate running totals for P&L tracking
        buy_volume = 0
        buy_value = 0
        sell_volume = 0
        sell_value = 0
        history_lines = []
        
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
            
            # Calculate average prices
            avg_buy_price = buy_value / buy_volume if buy_volume > 0 else 0
            avg_sell_price = sell_value / sell_volume if sell_volume > 0 else 0
            
            # Format trade line
            line = (
                f"Round {trade.round}: {role} {trade.quantity} @ "
                f"{MarketStateFormatter._format_currency(trade.price)} "
                f"(Total: {MarketStateFormatter._format_currency(trade.value)})"
            )
            history_lines.append(line)
        
        # Add summary section if there were trades
        if buy_volume > 0 or sell_volume > 0:
            history_lines.append("\nRecent Trading Summary:")
            if buy_volume > 0:
                history_lines.append(
                    f"Bought: {buy_volume} shares @ avg "
                    f"{MarketStateFormatter._format_currency(avg_buy_price)}"
                )
            if sell_volume > 0:
                history_lines.append(
                    f"Sold: {sell_volume} shares @ avg "
                    f"{MarketStateFormatter._format_currency(avg_sell_price)}"
                )
        
        return "\n".join(history_lines)

    @staticmethod
    def _prepare_redemption_context(fundamental_signal):
        """Prepare redemption information context"""
        redemption_value = fundamental_signal.metadata.get('redemption_value')
        periods_remaining = fundamental_signal.metadata.get('periods_remaining')
        
        if periods_remaining == "Infinite" or periods_remaining is None:
            redemption_text = "This market has an infinite time horizon. Shares will not be redeemed."
        else:
            rounds_left = int(periods_remaining)
            # Check if redemption_value is None and use fundamental value instead
            if redemption_value is None:
                redemption_value = 0
            
            if rounds_left > 0:
                redemption_text = f"At the end of the final round (in {rounds_left} rounds), all shares will be redeemed at ${redemption_value:.2f} per share."
            else:
                redemption_text = f"This is the final round. At the end of this round, all shares will be redeemed at ${redemption_value:.2f} per share."
        
        return {
            'redemption_text': redemption_text
        }