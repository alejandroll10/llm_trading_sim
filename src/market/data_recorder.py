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
                'committed_cash': round(state.committed_cash, 2),  # Cash locked in pending buy orders
                'total_cash': round(state.cash + state.committed_cash + state.dividend_cash, 2)  # Full total including committed
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
        """Record dividend model state and realizations including shock components"""
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

        # Get shock components from the last realization (if available)
        last_realization = dividend_state.get('last_realization')
        shock_data = {}
        if last_realization:
            shock_data = {
                'systematic_shock': round(last_realization.systematic_shock, 6),
                'style_shock': round(last_realization.style_shock, 6),
                'systematic_contribution': round(last_realization.systematic_contribution, 6),
                'style_contribution': round(last_realization.style_contribution, 6),
                'idiosyncratic_component': round(last_realization.idiosyncratic_component, 4),
                'base_component': round(last_realization.base_component, 4),
            }

        # Calculate actual total cash paid from payment_history (avoids timing issues)
        total_cash_paid = calculate_total_dividend_cash(
            self.agent_repository,
            round_number
        )

        record = {
            'round': round_number + 1,
            'timestamp': timestamp,
            'last_paid_dividend': last_paid,
            'base_dividend': model_info.base_dividend,
            'dividend_frequency': model_info.dividend_frequency,
            'next_payment_round': dividend_state.get('next_payment_round', 0),
            'should_pay': dividend_state.get('should_pay', False),
            'price': round(self.context.current_price, 2),
            'total_dividend_payment': round(total_cash_paid, 2) if dividend_state.get('should_pay', False) else 0,
            'style': dividend_state.get('style'),
            'systematic_beta': model_info.systematic_beta,
            'style_gamma': model_info.style_gamma,
        }
        # Add shock data if available
        record.update(shock_data)
        self.dividend_data.append(record)

    def record_multi_stock_dividends(
        self,
        round_number: int,
        dividends_by_stock: Dict[str, float],
        realizations_by_stock: Dict[str, Any] = None
    ):
        """Record per-stock dividend information for multi-stock scenarios.

        Args:
            round_number: Current simulation round
            dividends_by_stock: Dict mapping stock_id to dividend per-share amount paid
            realizations_by_stock: Optional dict mapping stock_id to DividendRealization objects
                                   (contains shock component breakdown)
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
            record = {
                'round': round_number + 1,
                'timestamp': timestamp,
                'stock_id': stock_id,
                'last_paid_dividend': round(dividend, 2),  # Per-share dividend for this stock
                'price': 0.0,  # Will be filled if needed
                'should_pay': dividend > 0,  # True if this stock paid a dividend
                'total_dividend_payment': round(stock_cash_paid[stock_id], 2),  # Actual cash paid from payment_history
                'is_multi_stock': True,
                'is_per_stock_detail': True  # Flag to distinguish from aggregate
            }

            # Add shock component breakdown if realization available
            if realizations_by_stock and stock_id in realizations_by_stock:
                realization = realizations_by_stock[stock_id]
                record.update({
                    'systematic_shock': round(realization.systematic_shock, 6),
                    'style_shock': round(realization.style_shock, 6),
                    'systematic_contribution': round(realization.systematic_contribution, 6),
                    'style_contribution': round(realization.style_contribution, 6),
                    'idiosyncratic_component': round(realization.idiosyncratic_component, 4),
                    'base_component': round(realization.base_component, 4),
                })

            self.dividend_data.append(record)

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

        # Save agent memory timeline (notes_to_self over time)
        self._save_agent_memory_timeline(data_path)

        # Save agent prompt evolution (self-modification history)
        self._save_agent_prompt_evolution(data_path)

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

    def _save_agent_memory_timeline(self, data_path: Path):
        """Export all agent memory notes to a separate CSV file for timeline analysis

        This creates a dedicated CSV file showing the evolution of each agent's
        memory notes over time, making it easy to analyze how agents learn and
        adapt their strategies.

        The notes are also saved in structured_decisions.csv, but this file
        provides a cleaner view focused solely on memory evolution.

        Args:
            data_path: Directory path where CSV should be saved
        """
        memory_data = []

        # Collect memory notes from all agents
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)

            # Check if agent has memory_notes attribute (only LLMAgents do)
            if hasattr(agent, 'memory_notes') and agent.memory_notes:
                agent_type_name = getattr(agent.agent_type, 'name', 'unknown')

                for round_num, note in agent.memory_notes:
                    memory_data.append({
                        'agent_id': agent_id,
                        'agent_type': agent_type_name,
                        'round': round_num,
                        'note': note,
                        'note_index': len([r for r, _ in agent.memory_notes if r <= round_num])  # Note number for this agent
                    })

        # Save to CSV if we have any memory notes
        if memory_data:
            memory_df = pd.DataFrame(memory_data)
            # Sort by round, then agent_id for chronological view
            memory_df = memory_df.sort_values(['round', 'agent_id'])
            memory_df.to_csv(data_path / 'agent_memory_timeline.csv', index=False)

            # Log summary (optional - loggers may not be configured)
            if hasattr(self, 'loggers') and self.loggers:
                # Use simulation logger if available
                try:
                    from services.logging_service import LoggingService
                    logger = LoggingService.get_logger('simulation')
                    logger.info(
                        f"Saved agent memory timeline: {len(memory_data)} notes from "
                        f"{len(set(m['agent_id'] for m in memory_data))} agents"
                    )
                except:
                    pass  # Logging is optional

    def _save_agent_prompt_evolution(self, data_path: Path):
        """Export all agent prompt evolution history to a separate CSV file

        This creates a dedicated CSV file showing how each agent's system prompt
        has evolved over time through self-modification, making it easy to analyze
        strategy evolution and emergent behaviors.

        Args:
            data_path: Directory path where CSV should be saved
        """
        prompt_data = []

        # Collect prompt history from all agents
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)

            # Check if agent has prompt_history attribute (only LLMAgents with SELF_MODIFY do)
            if hasattr(agent, 'prompt_history') and agent.prompt_history:
                agent_type_name = getattr(agent.agent_type, 'name', 'unknown')
                original_prompt = agent.agent_type.system_prompt

                for round_num, prompt in agent.prompt_history:
                    # Determine if this is a modification or the original
                    is_modification = round_num > 0

                    # Extract just the modification text if applicable
                    modification_text = ""
                    if is_modification and "[Strategy Update" in prompt:
                        # Find the latest modification block
                        mod_start = prompt.rfind("[Strategy Update")
                        modification_text = prompt[mod_start:]

                    prompt_data.append({
                        'agent_id': agent_id,
                        'agent_type': agent_type_name,
                        'round': round_num,
                        'is_modification': is_modification,
                        'modification_number': len([r for r, _ in agent.prompt_history if r <= round_num and r > 0]),
                        'modification_text': modification_text if is_modification else "(original)",
                        'full_prompt_length': len(prompt),
                        'full_prompt': prompt  # Store full prompt for analysis
                    })

        # Save to CSV if we have any prompt evolution data
        if prompt_data:
            prompt_df = pd.DataFrame(prompt_data)
            # Sort by agent_id, then round for chronological view per agent
            prompt_df = prompt_df.sort_values(['agent_id', 'round'])
            prompt_df.to_csv(data_path / 'agent_prompt_evolution.csv', index=False)

            # Log summary (optional)
            if hasattr(self, 'loggers') and self.loggers:
                try:
                    from services.logging_service import LoggingService
                    logger = LoggingService.get_logger('simulation')
                    num_modifications = sum(1 for p in prompt_data if p['is_modification'])
                    num_agents = len(set(p['agent_id'] for p in prompt_data))
                    logger.info(
                        f"Saved agent prompt evolution: {num_modifications} modifications from {num_agents} agents"
                    )
                except:
                    pass  # Logging is optional