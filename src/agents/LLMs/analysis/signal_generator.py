from typing import Dict, List
from market.information.information_types import InformationType, InformationSignal
from dataclasses import dataclass
from market.trade import Trade
from agents.LLMs.llm_agent import AgentContext
from datetime import datetime
import math
@dataclass
class MarketScenario:
    """Parameters for generating a market scenario"""
    # Core market values
    price: float = 100.0
    fundamental_value: float = 100.0
    volume: float = 1000.0
    
    # Scenario timing
    total_rounds: int = 10
    current_round: int = 1
    
    # Market structure
    spread_percent: float = 0.01
    order_book_depth: int = 5
    order_book_step: float = 0.001
    volume_per_level: float = 1000.0
    order_book_progression: str = 'linear'  # or 'exponential'
    
    # Payment schedules
    dividend_payment_interval: int = 3
    interest_payment_interval: int = 2
    dividend_payments_per_year: int = 12  # For monthly dividends
    
    # Signal reliability
    price_reliability: float = 1.0
    volume_reliability: float = 1.0
    fundamental_reliability: float = 1.0
    dividend_reliability: float = 1.0
    interest_reliability: float = 1.0
    order_book_reliability: float = 1.0
    
    # Order book parameters
    buy_order_discount: float = 0.02  # 2% below current price
    sell_order_premium: float = 0.02  # 2% above current price
    market_order_size_ratio: float = 0.5  # 50% of level volume
    limit_order_size_ratio: float = 0.75  # 75% of level volume
    
    # Rates and yields
    interest_rate: float = 0.05
    dividend_yield: float = 0.03
    compound_frequency: int = 4
   
    # History settings
    price_history_length: int = 5
    trade_history_length: int = 5
    signal_history_rounds: int = 5
    trade_pattern: str = 'alternate'  # or 'random', 'buyer_heavy', 'seller_heavy'
    
    # Agent parameters
    agent_id: str = "test_agent"
    
    # Agent state
    initial_cash: float = 10000.0
    initial_shares: int = 100
    reserved_cash_ratio: float = 0.1
    reserved_shares_ratio: float = 0.25
    
    # Market dynamics
    min_trade_size: int = 100
    max_trade_size: int = 1000
    trade_size_increment: int = 100
    
    # History variation parameters
    price_history_variation: float = 0.05  # 5% max variation
    price_history_pattern: str = 'constant'  # 'constant', 'trend_up', 'trend_down', 'oscillating'
    price_trend_strength: float = 0.01  # 1% change per period for trends
    price_oscillation_amplitude: float = 0.02  # 2% amplitude for oscillations
    
    @property
    def reserved_cash(self) -> float:
        return self.initial_cash * self.reserved_cash_ratio
    
    @property
    def reserved_shares(self) -> int:
        return int(self.initial_shares * self.reserved_shares_ratio)
    
    @property
    def available_cash(self) -> float:
        return self.initial_cash - self.reserved_cash
    
    @property
    def available_shares(self) -> int:
        return self.initial_shares - self.reserved_shares

    def calculate_historical_price(self, base_price: float, current_round: int, periods_back: int) -> float:
        """Calculate historical price using deterministic patterns
        
        Args:
            base_price: Current price to base history on
            current_round: Current round number
            periods_back: How many periods back from current round
        """
        historical_round = current_round - periods_back
        
        if historical_round <= 0:
            return base_price
            
        if self.price_history_pattern == 'constant':
            return base_price
            
        elif self.price_history_pattern == 'trend_up':
            # Price increases by trend_strength each period
            return base_price * (1 - periods_back * self.price_trend_strength)
            
        elif self.price_history_pattern == 'trend_down':
            # Price decreases by trend_strength each period
            return base_price * (1 + periods_back * self.price_trend_strength)
            
        elif self.price_history_pattern == 'oscillating':
            # Sinusoidal oscillation
            phase = (historical_round % 4) / 4.0  # Complete cycle every 4 rounds
            variation = math.sin(2 * math.pi * phase) * self.price_oscillation_amplitude
            return base_price * (1 + variation)
            
        return base_price  # Default to constant if pattern not recognized

class SignalGenerator:
    """Generates deterministic market signals for agent analysis"""
    
    def __init__(self, scenario: MarketScenario = MarketScenario()):
        self.scenario = scenario
        
    def _get_current_round(self, override_round: int = None) -> int:
        """Get current round number, respecting override if provided"""
        return override_round if override_round is not None else self.scenario.current_round
        
    def generate_test_signals(self, 
                            num_scenarios: int = 1,
                            round_number: int = None) -> List[Dict[InformationType, InformationSignal]]:
        """Generate test scenarios with exact parameters
        
        Args:
            num_scenarios: Number of scenarios to generate
            round_number: Override current round number (optional)
        """
        current_round = self._get_current_round(round_number)
        periods_remaining = self._calculate_remaining_periods(current_round)
        
        scenarios = []
        for _ in range(num_scenarios):
            # Use exact values from scenario
            price = self.scenario.price
            fundamental = self.scenario.fundamental_value
            
            # Calculate exact order book levels
            spread = price * self.scenario.spread_percent
            best_bid = price - spread/2
            best_ask = price + spread/2
            
            order_book = self._generate_order_book(best_bid, best_ask)
            
            signals = {
                InformationType.PRICE: InformationSignal(
                    type=InformationType.PRICE,
                    value=price,
                    reliability=self.scenario.price_reliability,
                    metadata={
                        'round': current_round,
                        'best_bid': best_bid,
                        'best_ask': best_ask
                    }
                ),
                
                InformationType.VOLUME: InformationSignal(
                    type=InformationType.VOLUME,
                    value=self.scenario.volume,
                    reliability=self.scenario.volume_reliability,
                    metadata={
                        'round': current_round,
                        'trade_history': []
                    }
                ),
                
                InformationType.ORDER_BOOK: InformationSignal(
                    type=InformationType.ORDER_BOOK,
                    value=order_book,
                    reliability=self.scenario.order_book_reliability,
                    metadata={
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'depth_levels': self.scenario.order_book_depth
                    }
                ),
                
                InformationType.FUNDAMENTAL: InformationSignal(
                    type=InformationType.FUNDAMENTAL,
                    value=fundamental,
                    reliability=self.scenario.fundamental_reliability,
                    metadata={
                        'round': current_round,
                        'periods_remaining': periods_remaining
                    }
                ),
                
                InformationType.DIVIDEND: InformationSignal(
                    type=InformationType.DIVIDEND,
                    value=price * self.scenario.dividend_yield / self.scenario.dividend_payments_per_year,
                    reliability=self.scenario.dividend_reliability,
                    metadata={
                        'yields': {
                            'expected': self.scenario.dividend_yield * 100,
                            'max': self.scenario.dividend_yield * 100,
                            'min': self.scenario.dividend_yield * 100,
                            'last': self.scenario.dividend_yield * 100
                        },
                        'last_paid_dividend': price * self.scenario.dividend_yield / self.scenario.dividend_payments_per_year,
                        'next_payment_round': min(
                            current_round + self.scenario.dividend_payment_interval, 
                            self.scenario.total_rounds
                        ),
                        'should_pay': False,
                        'variation': 0.0,
                        'probability': 100,
                        'destination': 'cash',
                        'tradeable': 'non-tradeable'
                    }
                ),
                
                InformationType.INTEREST: InformationSignal(
                    type=InformationType.INTEREST,
                    value=self.scenario.interest_rate,
                    reliability=self.scenario.interest_reliability,
                    metadata={
                        'compound_frequency': self.scenario.compound_frequency,
                        'last_payment': max(1, current_round - 1),
                        'next_payment_round': min(current_round + self.scenario.interest_payment_interval, self.scenario.total_rounds),
                        'interest_destination': 'main'
                    }
                )
            }
            self._validate_signal_metadata(signals)
            scenarios.append(signals)
            
        return scenarios

    def _generate_order_book(self, best_bid: float, best_ask: float) -> Dict:
        """Generate deterministic order book levels"""
        buy_levels = []
        sell_levels = []
        
        # Generate exact buy levels below best bid
        for i in range(self.scenario.order_book_depth):
            if self.scenario.order_book_progression == 'linear':
                buy_price = best_bid * (1 - i * self.scenario.order_book_step)
                sell_price = best_ask * (1 + i * self.scenario.order_book_step)
            else:  # exponential
                buy_price = best_bid * (1 - self.scenario.order_book_step) ** i
                sell_price = best_ask * (1 + self.scenario.order_book_step) ** i
            
            buy_levels.append({
                'price': buy_price, 
                'quantity': self.scenario.volume_per_level
            })
            
            sell_levels.append({
                'price': sell_price,
                'quantity': self.scenario.volume_per_level
            })
            
        return {
            'buy_levels': buy_levels,
            'sell_levels': sell_levels
        }

    def _generate_price_history(self, current_price: float, round_number: int) -> List[float]:
        """Generate deterministic price history based on scenario parameters"""
        history = []
        
        for i in range(self.scenario.price_history_length):
            periods_back = self.scenario.price_history_length - i
            historical_price = self.scenario.calculate_historical_price(
                base_price=current_price,
                current_round=round_number,
                periods_back=periods_back
            )
            history.append(historical_price)
            
        return history

    def generate_signal_history(self, round_number: int = None) -> Dict[int, Dict[InformationType, InformationSignal]]:
        """Generate historical signals for previous rounds"""
        current_round = self._get_current_round(round_number)
        history = {}
        
        start_round = max(1, current_round - self.scenario.signal_history_rounds)
        for historical_round in range(start_round, current_round):
            signals = self.generate_test_signals(num_scenarios=1, round_number=historical_round)[0]
            history[historical_round] = signals
            
        return history

    def generate_trade_history(self, round_number: int = None) -> List[Trade]:
        """Generate deterministic trade history
        
        Args:
            round_number: Current round number
        Returns:
            List of Trade objects representing historical trades
        """
        current_round = self._get_current_round(round_number)
        num_trades = min(self.scenario.trade_history_length, current_round - 1)
        
        trades = []
        start_round = current_round - num_trades
        for historical_round in range(start_round, current_round):
            # Use trade_pattern parameter
            if self.scenario.trade_pattern == 'alternate':
                is_buyer = historical_round % 2 == 0
            elif self.scenario.trade_pattern == 'buyer_heavy':
                is_buyer = historical_round % 3 != 0  # 2/3 chance of being buyer
            elif self.scenario.trade_pattern == 'seller_heavy':
                is_buyer = historical_round % 3 == 0  # 1/3 chance of being buyer
            else:  # 'random'
                is_buyer = hash(f"{self.scenario.agent_id}_{historical_round}") % 2 == 0  # deterministic random
                
            buyer_id = self.scenario.agent_id if is_buyer else f"other_agent_{historical_round}"
            seller_id = f"other_agent_{historical_round}" if is_buyer else self.scenario.agent_id
            
            # Generate deterministic trade size within bounds
            base_size = self.scenario.min_trade_size
            size_range = self.scenario.max_trade_size - self.scenario.min_trade_size
            steps = size_range // self.scenario.trade_size_increment
            step_index = hash(f"size_{historical_round}_{self.scenario.agent_id}") % (steps + 1)
            quantity = base_size + step_index * self.scenario.trade_size_increment
            
            # Generate deterministic order IDs
            buyer_order_id = f"buy_order_{historical_round}_{buyer_id}"
            seller_order_id = f"sell_order_{historical_round}_{seller_id}"
            
            # Create trade with deterministic timestamp
            trade = Trade(
                buyer_id=buyer_id,
                seller_id=seller_id,
                quantity=quantity,
                price=self.scenario.price,  # Use scenario price
                timestamp=datetime(2024, 1, 1, 12, 0, 0),  # Fixed timestamp for determinism
                round=historical_round,
                buyer_order_id=buyer_order_id,
                seller_order_id=seller_order_id
            )
            
            trades.append(trade)
        
        return trades

    def generate_test_context(self, round_number: int = None) -> AgentContext:
        """Generate a deterministic test context"""
        current_round = self._get_current_round(round_number)
        
        signal_history = self.generate_signal_history(round_number=current_round)
        trade_history = self.generate_trade_history(round_number=current_round)
        
        # Create deterministic outstanding orders
        outstanding_orders = {
            'buy': [
                {
                    'quantity': self.scenario.volume_per_level * self.scenario.limit_order_size_ratio,
                    'price': self.scenario.price * (1 - self.scenario.buy_order_discount)
                },
                {
                    'quantity': self.scenario.volume_per_level * self.scenario.market_order_size_ratio,
                    'price': None  # Market order
                }
            ],
            'sell': [
                {
                    'quantity': self.scenario.volume_per_level * self.scenario.limit_order_size_ratio,
                    'price': self.scenario.price * (1 + self.scenario.sell_order_premium)
                }
            ]
        }
        
        # Use scenario parameters for context values
        return AgentContext(
            agent_id=self.scenario.agent_id,
            cash=self.scenario.initial_cash,
            shares=self.scenario.initial_shares,
            available_cash=self.scenario.available_cash,
            available_shares=self.scenario.available_shares,
            outstanding_orders=outstanding_orders,
            signal_history=signal_history,
            trade_history=trade_history
        )

    def _validate_signal_metadata(self, signals: Dict[InformationType, InformationSignal]):
        """Validate that all required metadata fields are present"""
        
        # Price signal metadata
        price_metadata = signals[InformationType.PRICE].metadata
        assert 'round' in price_metadata, "Price signal missing 'round' metadata"
        assert 'best_bid' in price_metadata, "Price signal missing 'best_bid' metadata"
        assert 'best_ask' in price_metadata, "Price signal missing 'best_ask' metadata"
        
        # Fundamental signal metadata
        fundamental_metadata = signals[InformationType.FUNDAMENTAL].metadata
        assert 'periods_remaining' in fundamental_metadata, "Fundamental signal missing 'periods_remaining' metadata"
        assert 'round' in fundamental_metadata, "Fundamental signal missing 'round' metadata"
        
        # Dividend signal metadata
        dividend_metadata = signals[InformationType.DIVIDEND].metadata
        required_dividend_fields = [
            'yields', 'last_paid_dividend', 'next_payment_round', 
            'should_pay', 'variation', 'probability', 
            'destination', 'tradeable'
        ]
        for field in required_dividend_fields:
            assert field in dividend_metadata, f"Dividend signal missing '{field}' metadata"
        assert all(k in dividend_metadata['yields'] for k in ['expected', 'max', 'min'])
        
        # Interest signal metadata
        interest_metadata = signals[InformationType.INTEREST].metadata
        required_interest_fields = [
            'compound_frequency', 'last_payment',
            'next_payment_round', 'interest_destination'
        ]
        for field in required_interest_fields:
            assert field in interest_metadata, f"Interest signal missing '{field}' metadata"

    def _calculate_remaining_periods(self, current_round: int) -> int:
        return max(0, self.scenario.total_rounds - current_round)

    def _is_payment_round(self, current_round: int, interval: int) -> bool:
        return current_round % interval == 0