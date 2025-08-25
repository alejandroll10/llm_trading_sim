from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass

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

    def initialize_agent_structures(self):
        """Initialize data structures that depend on agents"""
        self.wealth_history = {
            agent_id: [] for agent_id in self.agent_repository.get_all_agent_ids()
        }

    def record_round_data(self, round_number: int, market_state: dict,
                         orders: List[dict], trades: List[dict], total_volume: float, dividends: float):
        """Record all data for a single round, including dividends"""
        timestamp = datetime.now().isoformat()
        # Calculate aggregate short interest before recording
        short_interest = sum(
            self.agent_repository.get_agent_state_snapshot(
                agent_id,
                self.context.current_price
            ).borrowed_shares
            for agent_id in self.agent_repository.get_all_agent_ids()
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
        self._record_agent_data(round_number, timestamp, dividends)
        
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
            'order_book': market_state['market_depth'],
            'last_trade_price': self.context.public_info['last_trade']['price'],
            'best_bid': self.context.public_info['order_book_state']['best_bid'],
            'best_ask': self.context.public_info['order_book_state']['best_ask'],
            'midpoint': self.context.public_info['order_book_state']['midpoint'],
            'short_interest': self.context.public_info.get('short_interest', 0)
        })

    def _record_market_data(self, round_number, market_state, trades, total_volume, timestamp):
        """Record market data"""
        self.market_data.append({
            'round': round_number + 1,
            'price': self.context.current_price,
            'fundamental_price': self.context.fundamental_price,
            'total_volume': total_volume,
            'num_trades': len(trades),
            'timestamp': timestamp,
            'price_fundamental_ratio': (self.context.current_price / 
                self.context.fundamental_price 
                if self.context.fundamental_price else 0),
            'best_bid': market_state['best_bid'],
            'best_ask': market_state['best_ask'],
            'market_depth': market_state['market_depth'],
            'short_interest': self.context.public_info.get('short_interest', 0)
        })

    def _record_trade_data(self, round_number, trades, timestamp):
        """Record trade data"""
        for trade in trades:
            self.trade_data.append({
                'round': round_number + 1,
                'buyer_id': trade.buyer_id,    # Assuming Trade object attributes
                'seller_id': trade.seller_id,
                'quantity': trade.quantity,
                'price': trade.price,
                'timestamp': timestamp
            })

    def _record_agent_data(self, round_number: int, timestamp: str, dividends: float):
        """Record agent data, including dividends"""
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

            dividends_received = round(state.net_shares * dividends, 2)
            
            # Update wealth history
            self.wealth_history[agent_id].append(current_wealth)
            
            # Record agent data
            self.agent_data.append({
                'round': round_number + 1,
                'agent_id': state.agent_id,
                'agent_type': state.agent_type,
                'cash': round(state.cash, 2),
                'shares': state.shares,
                'borrowed_shares': state.borrowed_shares,
                'net_shares': state.net_shares,
                'price': round(self.context.current_price, 2),
                'share_value': share_value,
                'total_value': current_wealth,
                'dividends_received': dividends_received,
                'timestamp': timestamp,
                'dividend_cash': round(state.dividend_cash, 2),
                'total_cash': round(state.cash + state.dividend_cash, 2)
            })

    def _record_order_data(self, round_number, orders, timestamp):
        """Record order data"""
        for order in orders:
            self.order_data.append({
                'round': round_number + 1,
                'agent_id': order.agent_id,  # Use attribute access
                'decision': order.side,      # Changed from decision to side
                'quantity': order.quantity,
                'price_limit': order.price,  # Changed from price_limit to price
                'order_type': order.order_type,  # Added order_type
                'timestamp': timestamp
            })

    def _record_dividend_data(self, round_number: int, market_state: dict, timestamp: str):
        """Record dividend model state and realizations"""
        if round_number == 0:
            return

        if not self.market_state_manager.dividend_service:
            raise ValueError("Dividend service not found. Recording dividend data for round {round_number}")
        
        dividend_state = market_state.get('dividend_state', {})
        if not dividend_state:
            raise ValueError("Dividend state not found")
        
        model_info = dividend_state['model']  # This is a DividendInfo object
        last_paid = round(dividend_state.get('last_paid_dividend', 0.0), 2)  # Round to 2 decimals
        
        # Calculate total shares through repository
        total_shares = sum(
            self.agent_repository.get_agent_state_snapshot(
                agent_id, 
                self.context.current_price
            ).shares
            for agent_id in self.agent_repository.get_all_agent_ids()
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
            'total_dividend_payment': round(
                last_paid * total_shares
                if dividend_state.get('should_pay', False) else 0,
                2
            )
        })

    def save_simulation_data(self):
        """Save all simulation data to files"""
        data_path = Path(self.data_dir)
        
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
        
        # Save dividend data
        dividend_df = pd.DataFrame(self.dividend_data)
        dividend_df.to_csv(data_path / 'dividend_data.csv', index=False)
        
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