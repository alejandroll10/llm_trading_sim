from typing import List, Dict
import pandas as pd
from sklearn.linear_model import LinearRegression
from .signal_generator import SignalGenerator, MarketScenario
from ..llm_agent import LLMAgent
from market.information.information_types import InformationType
from agents.LLMs.services.formatting_services import AgentContext

class AgentScenarioRunner:
    """Analyzes LLM agent decisions using synthetic data"""
    
    def __init__(self, agent: LLMAgent, scenario: MarketScenario = None):
        self.agent = agent
        self.scenario = scenario or MarketScenario()
        self.signal_generator = SignalGenerator(self.scenario)
        self.last_signals = None

    def run_single_trading_scenario(self) -> Dict:
        """Simulates a single trading scenario and captures the agent's decision.
        
        This method generates a complete trading environment including market signals,
        trading history, and agent context, then requests a trading decision from the
        LLM agent.
        
        Returns:
            Dict: A scenario result containing:
                - Market signals (price, fundamental value, volume, bid/ask prices)
                - Trading context (dividends, interest rates)
                - Agent's decision (type, quantity, price target)
                - Agent's state (cash, shares)
                - Decision metadata (reasoning, raw response, order details)
        """
        # Generate signals and context
        signals = self.signal_generator.generate_test_signals(
            num_scenarios=1, 
            round_number=self.scenario.current_round
        )[0]
        self.last_signals = signals
        
        signal_history = self.signal_generator.generate_signal_history(
            round_number=self.scenario.current_round
        )
        trade_history = self.signal_generator.generate_trade_history(
            round_number=self.scenario.current_round
        )
        
        # Create agent context
        agent_context = AgentContext(
            agent_id=self.agent.agent_id,
            cash=self.agent.cash,
            shares=self.agent.shares,
            available_cash=self.agent.available_cash,
            available_shares=self.agent.available_shares,
            outstanding_orders={'buy': [], 'sell': []},
            signal_history=signal_history,
            trade_history=trade_history
        )
        
        # Update agent's state
        self.agent.private_signals = signals
        self.agent.context = agent_context
        
        # Get decision
        decision_dict = self.agent.make_decision(
            market_state=signals,
            history=[],
            round_number=self.scenario.current_round
        )
        
        # Return compiled result with rounded values
        return {
            'round': self.scenario.current_round,
            'price': round(signals[InformationType.PRICE].value, 2),
            'fundamental': round(signals[InformationType.FUNDAMENTAL].value, 2),
            'volume': round(signals[InformationType.VOLUME].value, 2),
            'best_bid': round(signals[InformationType.PRICE].metadata.get('best_bid'), 2),
            'best_ask': round(signals[InformationType.PRICE].metadata.get('best_ask'), 2),
            'expected_dividend': round(signals[InformationType.DIVIDEND].value, 2),
            'interest_rate': round(signals[InformationType.INTEREST].value, 2),
            'decision_type': decision_dict['decision'],
            'quantity': decision_dict['quantity'],
            'price_limit': round(decision_dict.get('price_limit', 0), 2) if decision_dict.get('price_limit') is not None else None,
            'agent_cash': round(agent_context.cash, 2),
            'agent_shares': agent_context.shares,
            'reasoning': decision_dict['reasoning'],
            'raw_response': decision_dict.get('raw_response', ''),
            'order_type': decision_dict.get('order_type'),
            'replace_decision': decision_dict.get('replace_decision', 'Replace')
        }