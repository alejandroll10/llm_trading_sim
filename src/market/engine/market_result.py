from dataclasses import dataclass
from typing import List
from datetime import datetime
from market.trade import Trade

@dataclass
class MarketResult:
    """
    Encapsulates the results of a market matching round
    """
    trades: List[Trade]  # List of executed trades
    price: float       # New market price after matching
    volume: float      # Total trading volume in this round
    
    def __post_init__(self):
        """Calculate additional metrics after initialization"""
        self.timestamp = datetime.now()
        self.num_trades = len(self.trades)
        
        # Calculate VWAP using dot notation for Trade objects
        if self.volume > 0:
            self.vwap = sum(t.price * t.quantity for t in self.trades) / self.volume
        else:
            self.vwap = self.price
            
    def to_dict(self):
        """Convert result to dictionary format"""
        return {
            'timestamp': self.timestamp,
            'price': self.price,
            'volume': self.volume,
            'num_trades': self.num_trades,
            'vwap': self.vwap,
            'trades': [t.to_dict() for t in self.trades]
        }