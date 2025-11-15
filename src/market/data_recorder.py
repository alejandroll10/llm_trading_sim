from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass

from services.short_interest_calculator import calculate_short_interest
from services.dividend_calculator import (
    calculate_agent_dividends_received,
    calculate_total_dividend_cash,
    calculate_multi_stock_dividend_cash
)

@dataclass
class AgentRecordData:
    agent_id: str
    agent_type: str
    cash: float
    shares: int
    borrowed_shares: int
    net_shares: int
    share_value: float
    total_value: float
    dividends_received: float

class DataRecorder:
    def __init__(self,
                 context,
                 agent_repository,
                 market_state_manager,
                 loggers,
                 data_dir: Path):
        # Core dependencies
        self.context = context
        self.agent_repository = agent_repository
        self.market_state_manager = market_state_manager
        self.loggers = loggers
        self.data_dir = data_dir
        
        # Data structures
        self.history: List[Dict[str, Any]] = []
        self.market_data: List[Dict[str, Any]] = []
        self.trade_data: List[Dict[str, Any]] = []
        self.agent_data: List[Dict[str, Any]] = []
        self.order_data: List[Dict[str, Any]] = []
        self.wealth_history: Dict[int, List[float]] = {}
        self.dividend_data: List[Dict[str, Any]] = []
        self.social_messages: List[Dict[str, Any]] = []
        self.stock_positions: List[Dict[str, Any]] = []  # NEW: Per-stock position tracking

    def initialize_agent_structures(self):
        """Initialize data structures that depend on agents"""
        self.wealth_history = {
            agent_id: [] for agent_id in self.agent_repository.get_all_agent_ids()
        }

    def record_round_data(self, round_number: int, market_state: dict,
                         orders: List[dict], trades: List[dict], total_volume: float, dividends: float,
                         dividends_by_stock: Dict[str, float] = None):
        """Record all data for a single round, including dividends

        Args:
            round_number: Current simulation round
            market_state: Market state dictionary
            orders: List of orders
            trades: List of trades
            total_volume: Total trading volume
            dividends: Aggregate dividend per share (single-stock) or sum of all dividends (multi-stock)
            dividends_by_stock: Dict mapping stock_id to per-share dividend (multi-stock only)
        """
        timestamp = datetime.now().isoformat()
        # Calculate aggregate short interest before recording
        short_interest = calculate_short_interest(
            self.agent_repository,
            self.context.current_price
        )
        self.context.update_short_interest(short_interest)

        # Record market history
        self._record_market_history(round_number, market_state, orders,
                                  trades, total_volume)

        # Record market data
        self._record_market_data(round_number, market_state, trades,
                               total_volume, timestamp)

        # Record trade data
        self._record_trade_data(round_number, trades, timestamp)

        # Record agent data
        self._record_agent_data(round_number, timestamp, dividends, dividends_by_stock)

        # Record stock positions (multi-stock only)
        self._record_stock_positions(round_number, market_state, timestamp)

        # Record order data
        self._record_order_data(round_number, orders, timestamp)

        # Add dividend data recording
        self._record_dividend_data(round_number, market_state, timestamp)

    def _record_market_history(self, round_number, market_state, orders, trades, total_volume):
        """Record market history data"""
        order_dicts = [{
            'agent_id': order.agent_id,
            'side': order.side,
            'quantity': order.quantity,
            'price': order.price,
            'order_type': order.order_type,
            'order_id': order.order_id
        } for order in orders]

        # Handle multi-stock vs single-stock market_state
        if market_state.get('is_multi_stock'):
            # Multi-stock: Get market_depth from first stock for backwards compatibility
            first_stock_state = list(market_state['stocks'].values())[0] if market_state['stocks'] else {}
            order_book = first_stock_state.get('market_depth', {})
        else:
            # Single-stock: Original behavior
            order_book = market_state.get('market_depth', {})

        self.history.append({
            'round': round_number + 1,
            'price': self.context.current_price,
            'fundamental_price': self.context.fundamental_price,
            'total_volume': total_volume,
            'trades': trades,
            'orders': order_dicts,
            'price_fundamental_ratio': (
                self.context.current_price / self.context.fundamental_price
                if self.context.fundamental_price else 0
            ),
            'order_book': order_book,
            'last_trade_price': self.context.public_info['last_trade']['price'],
            'best_bid': self.context.public_info['order_book_state']['best_bid'],
            'best_ask': self.context.public_info['order_book_state']['best_ask'],
            'midpoint': self.context.public_info['order_book_state']['midpoint'],
            'short_interest': self.context.public_info.get('short_interest', 0)
        })

    def _record_market_data(self, round_number, market_state, trades, total_volume, timestamp):
        """Record market data - one row per stock in multi-stock mode"""
        if market_state.get('is_multi_stock'):
            # Multi-stock: Record each stock separately
            for stock_id, stock_state in market_state['stocks'].items():
                # Calculate volume for this specific stock
                stock_trades = [t for t in trades if t.stock_id == stock_id]
                stock_volume = sum(t.quantity for t in stock_trades)

                self.market_data.append({
                    'round': round_number + 1,
                    'stock_id': stock_id,
                    'price': stock_state['price'],
                    'fundamental_price': stock_state['fundamental_price'],
                    'total_volume': stock_volume,
                    'num_trades': len(stock_trades),
                    'timestamp': timestamp,
                    'price_fundamental_ratio': (
                        stock_state['price'] / stock_state['fundamental_price']
                        if stock_state['fundamental_price'] else 0
                    ),
                    'best_bid': stock_state.get('best_bid'),
                    'best_ask': stock_state.get('best_ask'),
                    'market_depth': stock_state.get('market_depth', {}),
                    'short_interest': stock_state.get('short_interest', 0)
                })
        else:
            # Single-stock: Original behavior
            self.market_data.append({
                'round': round_number + 1,
                'stock_id': 'DEFAULT_STOCK',
                'price': self.context.current_price,
                'fundamental_price': self.context.fundamental_price,
                'total_volume': total_volume,
                'num_trades': len(trades),
                'timestamp': timestamp,
                'price_fundamental_ratio': (self.context.current_price /
                    self.context.fundamental_price
                    if self.context.fundamental_price else 0),
                'best_bid': market_state.get('best_bid'),
                'best_ask': market_state.get('best_ask'),
                'market_depth': market_state.get('market_depth', {}),
                'short_interest': self.context.public_info.get('short_interest', 0)
            })

    def _record_trade_data(self, round_number, trades, timestamp):
        """Record trade data"""
        for trade in trades:
            self.trade_data.append({
                'round': round_number + 1,
                'buyer_id': trade.buyer_id,
                'seller_id': trade.seller_id,
                'stock_id': trade.stock_id,  # NEW: Multi-stock support
                'quantity': trade.quantity,
                'price': trade.price,
                'timestamp': timestamp
            })

    def _record_agent_data(self, round_number: int, timestamp: str, dividends: float,
                          dividends_by_stock: Dict[str, float] = None):
        """Record agent data, including dividends

        Args:
            round_number: Current simulation round
            timestamp: Recording timestamp
            dividends: Aggregate dividend (single-stock) or sum (multi-stock)
            dividends_by_stock: Per-stock dividends for multi-stock scenarios
        """
        for agent_id in self.agent_repository.get_all_agent_ids():
            # Get agent state through repository
            state = self.agent_repository.get_agent_state_snapshot(
                agent_id,
                self.context.current_price
            )

            # Calculate values
            share_value = round(state.net_shares * self.context.current_price, 2)

            # Use the wealth already calculated by the agent
            current_wealth = round(state.wealth, 2)

            # Calculate dividends received based on actual payments from payment history
            # This avoids timing issues where positions change between dividend payment and recording
            agent = self.agent_repository.get_agent(agent_id)
            dividends_received = calculate_agent_dividends_received(
                agent,
                round_number,
                dividends,
                dividends_by_stock,
                state.net_shares
            )
            
            # Update wealth history
            self.wealth_history[agent_id].append(current_wealth)
            
            # Record agent data
            self.agent_data.append({
                'round': round_number + 1,
                'agent_id': state.agent_id,
                'agent_type': state.agent_type,
                'cash': round(state.cash, 2),
                'available_shares': state.shares,  # Shares available (not in pending orders)
                'committed_shares': state.committed_shares,  # Shares locked in pending orders
                'total_shares': state.total_shares,  # available_shares + committed_shares
                'borrowed_shares': state.borrowed_shares,  # Shares borrowed from lending pool
                'net_shares': state.net_shares,  # total_shares - borrowed_shares (economic position)
                'borrowed_cash': round(state.borrowed_cash, 2),  # Cash borrowed via leverage
                'leverage_interest_paid': round(state.leverage_interest_paid, 2),  # Cumulative interest on borrowed cash
                'price': round(self.context.current_price, 2),
                'share_value': share_value,
                'total_value': current_wealth,
                'dividends_received': dividends_received,
                'timestamp': timestamp,
                'dividend_cash': round(state.dividend_cash, 2),
                'total_cash': round(state.cash + state.dividend_cash, 2)
            })

    def _record_stock_positions(self, round_number: int, market_state: dict, timestamp: str):
        """Record per-stock positions for each agent in multi-stock mode"""
        if not market_state.get('is_multi_stock'):
            return  # Only for multi-stock simulations

        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)

            # Record position for each stock
            for stock_id, stock_state in market_state['stocks'].items():
                price = stock_state['price']

                # Get positions for this specific stock
                available_shares = agent.positions.get(stock_id, 0)
                committed_shares = agent.committed_positions.get(stock_id, 0)
                borrowed_shares = agent.borrowed_positions.get(stock_id, 0)
                total_shares = available_shares + committed_shares
                net_shares = total_shares - borrowed_shares

                # Calculate value
                share_value = round(net_shares * price, 2)

                self.stock_positions.append({
                    'round': round_number + 1,
                    'agent_id': agent_id,
                    'stock_id': stock_id,
                    'price': round(price, 2),
                    'available_shares': available_shares,
                    'committed_shares': committed_shares,
                    'total_shares': total_shares,
                    'borrowed_shares': borrowed_shares,
                    'net_shares': net_shares,
                    'share_value': share_value,
                    'timestamp': timestamp
                })

    def _record_order_data(self, round_number, orders, timestamp):
        """Record order data"""
        for order in orders:
            self.order_data.append({
                'round': round_number + 1,
                'agent_id': order.agent_id,
                'stock_id': order.stock_id,  # NEW: Multi-stock support
                'decision': order.side,
                'quantity': order.quantity,
                'price_limit': order.price,
                'order_type': order.order_type,
                'timestamp': timestamp
            })

    def _record_dividend_data(self, round_number: int, market_state: dict, timestamp: str):
        """Record dividend model state and realizations"""
        if round_number == 0:
            return

        # Skip dividend recording for multi-stock (not yet implemented)
        if market_state.get('is_multi_stock'):
            return

        if not self.market_state_manager.dividend_service:
            raise ValueError("Dividend service not found. Recording dividend data for round {round_number}")

        dividend_state = market_state.get('dividend_state', {})
        if not dividend_state:
            raise ValueError("Dividend state not found")

        model_info = dividend_state['model']  # This is a DividendInfo object
        last_paid = round(dividend_state.get('last_paid_dividend', 0.0), 2)  # Round to 2 decimals

        # Calculate actual total cash paid from payment_history (avoids timing issues)
        total_cash_paid = calculate_total_dividend_cash(
            self.agent_repository,
            round_number
        )

        self.dividend_data.append({
            'round': round_number + 1,
            'timestamp': timestamp,
            'last_paid_dividend': last_paid,
            'base_dividend': model_info.base_dividend,
            'dividend_frequency': model_info.dividend_frequency,
            'next_payment_round': dividend_state.get('next_payment_round', 0),
            'should_pay': dividend_state.get('should_pay', False),
            'price': round(self.context.current_price, 2),
            'total_dividend_payment': round(total_cash_paid, 2) if dividend_state.get('should_pay', False) else 0
        })

    def record_multi_stock_dividends(self, round_number: int, dividends_by_stock: Dict[str, float]):
        """Record per-stock dividend information for multi-stock scenarios.

        Args:
            round_number: Current simulation round
            dividends_by_stock: Dict mapping stock_id to dividend per-share amount paid
        """
        timestamp = datetime.now().isoformat()
        total_aggregated_dividend = sum(dividends_by_stock.values())

        # Calculate actual total cash paid using payment_history (avoids timing issues)
        total_cash_paid_aggregate, stock_cash_paid = calculate_multi_stock_dividend_cash(
            self.agent_repository,
            round_number,
            list(dividends_by_stock.keys())
        )

        # Record aggregate dividend (for backwards compatibility with save_simulation_data)
        # Use 'last_paid_dividend' field name to match single-stock format
        self.dividend_data.append({
            'round': round_number + 1,
            'timestamp': timestamp,
            'last_paid_dividend': round(total_aggregated_dividend, 2),  # Sum of per-share dividends across stocks
            'price': 0.0,  # Not applicable for aggregated multi-stock
            'should_pay': total_aggregated_dividend > 0,  # True if any dividend was paid
            'total_dividend_payment': round(total_cash_paid_aggregate, 2),  # Actual total cash paid from payment_history
            'is_multi_stock': True,
            'num_stocks': len(dividends_by_stock)
        })

        # Record per-stock dividends for detailed analytics
        for stock_id, dividend in dividends_by_stock.items():
            self.dividend_data.append({
                'round': round_number + 1,
                'timestamp': timestamp,
                'stock_id': stock_id,
                'last_paid_dividend': round(dividend, 2),  # Per-share dividend for this stock
                'price': 0.0,  # Will be filled if needed
                'should_pay': dividend > 0,  # True if this stock paid a dividend
                'total_dividend_payment': round(stock_cash_paid[stock_id], 2),  # Actual cash paid from payment_history
                'is_multi_stock': True,
                'is_per_stock_detail': True  # Flag to distinguish from aggregate
            })

    def record_social_message(self, round_number: int, agent_id: str, message: str):
        """Record a social media post from an agent"""
        self.social_messages.append({
            'round': round_number,
            'agent_id': agent_id,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

    def save_simulation_data(self):
        """Save all simulation data to files"""
        data_path = Path(self.data_dir)

        # Import messages from MessagingService before saving
        from services.messaging_service import MessagingService
        all_messages = MessagingService.get_all_messages()
        if all_messages:
            # Add timestamps to messages
            import datetime
            for msg in all_messages:
                msg['timestamp'] = datetime.datetime.now().isoformat()
            self.social_messages.extend(all_messages)

        # Save market data
        market_df = pd.DataFrame(self.market_data)
        market_df.to_csv(data_path / 'market_data.csv', index=False)

        # Save trade data
        trade_df = pd.DataFrame(self.trade_data)
        trade_df.to_csv(data_path / 'trade_data.csv', index=False)

        # Save agent data
        agent_df = pd.DataFrame(self.agent_data)
        agent_df.to_csv(data_path / 'agent_data.csv', index=False)

        # Save order data
        order_df = pd.DataFrame(self.order_data)
        order_df.to_csv(data_path / 'order_data.csv', index=False)

        # Save wealth history
        wealth_df = pd.DataFrame(self.wealth_history)
        wealth_df.to_csv(data_path / 'wealth_history.csv', index=False)

        # Save stock positions (multi-stock)
        if self.stock_positions:
            stock_positions_df = pd.DataFrame(self.stock_positions)
            stock_positions_df.to_csv(data_path / 'stock_positions.csv', index=False)

        # Save dividend data
        dividend_df = pd.DataFrame(self.dividend_data)
        dividend_df.to_csv(data_path / 'dividend_data.csv', index=False)

        # Save social messages
        if self.social_messages:
            social_df = pd.DataFrame(self.social_messages)
            social_df.to_csv(data_path / 'social_messages.csv', index=False)

        # Create a summary statistics file with safe dividend access
        dividend_model = self.market_state_manager.dividend_model
        
        # Safe dividend calculations
        dividend_values = [d['last_paid_dividend'] for d in self.dividend_data if d['last_paid_dividend'] is not None]
        dividend_yields = [
            (d['last_paid_dividend'] / d['price']) * 100 
            for d in self.dividend_data 
            if d['last_paid_dividend'] is not None and d['price'] > 0
        ]

        summary_data = {
            'avg_price': np.mean([d['price'] for d in self.market_data]),
            'price_volatility': np.std([d['price'] for d in self.market_data]),
            'avg_volume': np.mean([d['total_volume'] for d in self.market_data]),
            'total_trades': sum(d['num_trades'] for d in self.market_data),
            'avg_price_fundamental_ratio': np.mean([d['price_fundamental_ratio'] for d in self.market_data]),
            'avg_dividend': np.mean(dividend_values) if dividend_values else 0.0,
            'total_dividends_paid': sum(
                d['total_dividend_payment'] for d in self.dividend_data
                if d['should_pay']  # Changed from pay_dividends to should_pay
            ),
            'dividend_frequency': dividend_model.dividend_frequency if dividend_model else None,  # Access attribute directly
            'dividend_payments_count': sum(1 for d in self.dividend_data if d['should_pay']),  # Changed from pay_dividends to should_pay
            'avg_dividend_yield': np.mean(dividend_yields) if dividend_yields else 0.0,
        }
        
        with open(data_path / 'summary_statistics.json', 'w') as f:
            json.dump(summary_data, f, indent=4)