from market.orders.order import Order
from dataclasses import dataclass
from datetime import datetime
from market.trade import Trade
from typing import Dict, List, Any

@dataclass
class MarketTruth:
    """Container for true market state"""
    fundamental_price: float

@dataclass
class MarketHistory:
    """Container for historical market information"""
    fundamental_prices: List[float]
    dividends_paid: List[float]
    interest_paid: List[float]
    trade_history: List[Dict]
    quote_history: List[Dict]
    short_interest: List[float]
    borrow_fees_paid: List[float]

class SimulationContext:
    """
    Manages the global state of the market simulation.
    
    Responsibilities:
    - Stores fundamental values
    - Maintains market history
    - Provides public market information
    
    Does NOT:
    - Generate information signals (InformationService)
    - Manage agent states (AgentRepository)
    - Handle market mechanics (MarketStateManager)
    """
    def __init__(self, 
                 num_rounds: int,
                 initial_price: float,
                 fundamental_price: float,
                 redemption_value: float,
                 transaction_cost: float,
                 round_number: int = 0,
                 logger = None,
                 infinite_rounds: bool = False):
        
        self.logger = logger
        
        self.fundamental_price = fundamental_price
        self.redemption_value = redemption_value
        # Historical information
        self.market_history = MarketHistory(
            fundamental_prices=[fundamental_price],
            dividends_paid=[],
            interest_paid=[],
            trade_history=[],
            quote_history=[],
            short_interest=[0],
            borrow_fees_paid=[]
        )
        
        # Public market information (observable by all)
        self.public_info = {
            'current_price': initial_price,
            'transaction_cost': transaction_cost,
            'round_number': round_number,
            'last_trade': {
                'price': initial_price,
                'volume': 0,
                'timestamp': datetime.now().isoformat(),
                'round': 0
            },
            'order_book_state': {
                'best_bid': None,
                'best_ask': None,
                'midpoint': None,
                'aggregated_levels': {'buy_levels': [], 'sell_levels': []}
            },
            'trade_history': [],
            'short_interest': 0
        }
        
        # Simulation parameters
        self.infinite_rounds = infinite_rounds
        self._num_rounds = num_rounds
        self.initial_price = initial_price

    def update_short_interest(self, short_interest: float):
        """Update aggregate short interest and record history"""
        self.public_info['short_interest'] = short_interest
        self.market_history.short_interest.append(short_interest)
    
    def record_dividend_payment(self, amount: float, round_number: int):
        """Record dividend payment"""
        self.market_history.dividends_paid.append({
            'round': round_number,
            'amount': amount,
            'timestamp': datetime.now().isoformat()
        })
    
    def record_interest_payment(self, amount: float, round_number: int):
        """Record interest payment"""
        self.market_history.interest_paid.append({
            'round': round_number,
            'amount': amount,
            'timestamp': datetime.now().isoformat()
        })

    def record_borrow_fee_payment(self, amount: float, round_number: int):
        """Record borrow fee payment"""
        self.market_history.borrow_fees_paid.append({
            'round': round_number,
            'amount': amount,
            'timestamp': datetime.now().isoformat()
        })

    # Information access
    def get_public_info(self):
        """Return only public market information"""
        return self.public_info.copy()

    @property
    def current_price(self) -> float:
        """Get the current market price"""
        return self.public_info['current_price']
    
    @current_price.setter
    def current_price(self, value: float):
        """Set the current market price"""
        self.public_info['current_price'] = value
        self.public_info['last_trade']['price'] = value  # Update last trade price too
    
    @property
    def round_number(self) -> int:
        """Get the current round number"""
        return self.public_info['round_number']
    
    @round_number.setter
    def round_number(self, value: int):
        """Set the current round number"""
        self.public_info['round_number'] = value
    
    def update_trade_info(self, trade_price: float, trade_volume: int, round_number: int):
        """Update last trade information"""
        self.public_info['last_trade'].update({
            'price': trade_price,
            'volume': trade_volume,
            'timestamp': datetime.now().isoformat(),
            'round': round_number
        })
    
    def add_trade(self, trade: Trade):
        """Add trade to history and update last trade info"""
        trade_data = trade.to_dict()
        self.public_info['trade_history'].append(trade_data)
        
        self.update_trade_info(
            trade_price=trade.price,
            trade_volume=trade.quantity,
            round_number=self.round_number
        )
    
    def update_public_info(self, round_number: int, last_volume: float):
        """Update public market information"""
        self.public_info.update({
            'round_number': round_number,
            'last_trade': {
                'volume': last_volume,
                'round': round_number,
                **self.public_info['last_trade']  # Preserve other fields
            }
        })
