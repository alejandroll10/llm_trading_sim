from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .signal_generator import MarketScenario
from ..llm_agent import LLMAgent
from market.information.information_types import InformationType
from agents.LLMs.services.formatting_services import AgentContext
from agents.LLMs.analysis.agent_scenario_runner import AgentScenarioRunner
import numpy as np
from visualization.plot_config import STANDARD_FIGSIZE

class PriceFundamentalAnalyzer:
    """Analyzes agent behavior across different price/fundamental ratios"""
    
    def __init__(self, agent: LLMAgent, base_scenario: MarketScenario = None, repeats_per_ratio: int = 1):
        self.agent = agent
        # Use default scenario if none provided
        self.base_scenario = base_scenario or MarketScenario()
        self.repeats_per_ratio = repeats_per_ratio
        
    def run_ratio_analysis(
        self, 
        ratios: List[float],
        save_dir: Path,
        log_callback = None,
        repeats_per_ratio: int = 1
    ) -> Dict:
        """Run analysis across different price/fundamental ratios
        
        Args:
            ratios: List of P/F ratios to test
            save_dir: Directory to save results
            log_callback: Optional callback for logging conversations
            repeats_per_ratio: Number of times to test each ratio (default: 1)
        """
        all_results = []
        
        for ratio in ratios:
            # Create scenario with specific P/F ratio
            scenario = self._create_ratio_scenario(ratio)
            analyzer = AgentScenarioRunner(self.agent, scenario)
            
            # Generate signals ONCE per ratio
            signals = analyzer.signal_generator.generate_test_signals(num_scenarios=1)[0]
            analyzer.last_signals = signals
            
            # Get multiple decisions for the same ratio/signals
            for repeat in range(repeats_per_ratio):
                # Reset agent's state but keep same signals
                self.agent.private_signals = signals
                
                # Get decision
                result = analyzer.run_single_trading_scenario()
                result['current_ratio'] = ratio
                result['repeat_num'] = repeat + 1  # Track which repeat this is
                
                # Log if needed
                if log_callback:
                    context = self.agent.prepare_context_llm()
                    prompt = self.agent.agent_type.user_prompt_template.format(**context)
                    response = result.get('raw_response', 'No response available')
                    log_callback(repeat + 1, ratio, prompt, response)
                
                all_results.append(result)
        
        # Combine all results
        combined_df = pd.DataFrame(all_results)
        
        # Round current_ratio to 2 decimals for cleaner display
        combined_df['current_ratio'] = combined_df['current_ratio'].round(2)
        
        # Analyze and save
        ratio_analysis = self._analyze_ratio_patterns(combined_df)
        self._plot_ratio_patterns(combined_df, save_dir)  # Add plotting call
        combined_df.to_csv(save_dir / "ratio_decisions.csv")
        
        return {
            'dataframe': combined_df,
            'comparative_analysis': ratio_analysis,
            'results_dir': save_dir
        }
    
    def _create_ratio_scenario(self, current_ratio: float) -> MarketScenario:
        """Create a scenario with a specific price/fundamental ratio
        
        Args:
            current_ratio: Desired price/fundamental ratio
        """
        # Keep fundamental value constant and adjust price
        fundamental_value = self.base_scenario.fundamental_value
        price = round(fundamental_value * current_ratio, 2)  # Round to 2 decimals
        
        return MarketScenario(
            # Core market values
            price=price,
            fundamental_value=fundamental_value,
            volume=self.base_scenario.volume,
            
            # Scenario timing - reset for each ratio test
            total_rounds=self.base_scenario.total_rounds,
            current_round=self.base_scenario.current_round,  # Always start at round 1
            
            # Copy all other parameters from base scenario
            spread_percent=self.base_scenario.spread_percent,
            order_book_depth=self.base_scenario.order_book_depth,
            order_book_step=self.base_scenario.order_book_step,
            volume_per_level=self.base_scenario.volume_per_level,
            order_book_progression=self.base_scenario.order_book_progression,
            
            # Payment schedules
            dividend_payment_interval=self.base_scenario.dividend_payment_interval,
            interest_payment_interval=self.base_scenario.interest_payment_interval,
            dividend_payments_per_year=self.base_scenario.dividend_payments_per_year,
            
            # Keep all reliabilities at 1.0 for deterministic behavior
            price_reliability=1.0,
            volume_reliability=1.0,
            fundamental_reliability=1.0,
            dividend_reliability=1.0,
            interest_reliability=1.0,
            order_book_reliability=1.0,
            
            # History settings
            price_history_pattern='constant',  # Keep price history constant
            trade_pattern=self.base_scenario.trade_pattern,
            price_history_length=self.base_scenario.price_history_length,
            trade_history_length=self.base_scenario.trade_history_length,
            signal_history_rounds=self.base_scenario.signal_history_rounds
        )
    
    def _analyze_ratio_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns across all ratios"""
        analysis = {
            'decision_distributions': {},
            'avg_quantities': {},
            'order_types': {},
            'price_limits': {},
            'replace_decisions': {},
            'reasoning_summary': {},
            'summary': {}
        }
        
        # Analyze patterns for each ratio
        for ratio in df['current_ratio'].unique():
            ratio_df = df[df['current_ratio'] == ratio]
            
            # Basic decision distribution
            analysis['decision_distributions'][ratio] = (
                ratio_df['decision_type'].value_counts(normalize=True).to_dict()
            )
            
            # Store reasoning for each decision at this ratio
            analysis['reasoning_summary'][ratio] = {
                'decisions': [],
                'raw_responses': []
            }
            for _, row in ratio_df.iterrows():
                analysis['reasoning_summary'][ratio]['decisions'].append({
                    'decision_type': row['decision_type'],
                    'quantity': row['quantity'],
                    'order_type': row['order_type'],
                    'price_limit': row['price_limit'],
                    'replace_decision': row['replace_decision'],
                    'reasoning': row['reasoning']
                })
                if 'raw_response' in row:
                    analysis['reasoning_summary'][ratio]['raw_responses'].append(row['raw_response'])
            
            # Detailed trade analysis
            active_trades = ratio_df[ratio_df['decision_type'].isin(['Buy', 'Sell'])]
            if not active_trades.empty:
                # Order types
                analysis['order_types'][ratio] = (
                    active_trades['order_type'].value_counts(normalize=True).to_dict()
                )
                
                # Price limits for limit orders - separate by buy/sell
                limit_orders = active_trades[active_trades['order_type'] == 'limit']
                if not limit_orders.empty:
                    # Separate analysis for buy and sell orders
                    analysis['price_limits'][ratio] = {
                        'buy': {
                            'mean': limit_orders[limit_orders['decision_type'] == 'Buy']['price_limit'].mean(),
                            'std': limit_orders[limit_orders['decision_type'] == 'Buy']['price_limit'].std(),
                            'count': len(limit_orders[limit_orders['decision_type'] == 'Buy'])
                        },
                        'sell': {
                            'mean': limit_orders[limit_orders['decision_type'] == 'Sell']['price_limit'].mean(),
                            'std': limit_orders[limit_orders['decision_type'] == 'Sell']['price_limit'].std(),
                            'count': len(limit_orders[limit_orders['decision_type'] == 'Sell'])
                        }
                    }
                
                # Replace decisions
                analysis['replace_decisions'][ratio] = (
                    active_trades['replace_decision'].value_counts(normalize=True).to_dict()
                )
            
            # Quantities by decision type
            qty_by_decision = ratio_df.groupby('decision_type').agg({
                'quantity': ['mean', 'std', 'count']
            }).to_dict()
            analysis['avg_quantities'][ratio] = qty_by_decision
        
        # Overall summary
        analysis['summary'] = {
            'total_decisions': len(df),
            'decisions_by_ratio': df.groupby('current_ratio').size().to_dict(),
            'decision_types': df['decision_type'].value_counts().to_dict(),
            'order_types': df['order_type'].value_counts().to_dict(),
            'replace_decisions': df['replace_decision'].value_counts().to_dict()
        }
        
        return analysis
    
    def _create_plot(self, df: pd.DataFrame, plot_type: str, colors: Dict) -> None:
        """Create standardized plots with common parameters"""
        plt.figure(figsize=STANDARD_FIGSIZE)
        
        if plot_type == 'price_analysis':
            # Plot zero line as reference (current price)
            plt.axhline(y=0, color='k', linestyle='--', label='Current Price', alpha=0.5)
            
            # Plot limit orders as percentage difference from current price
            limit_orders = df[df['order_type'] == 'limit']
            for decision_type in ['Buy', 'Sell']:
                decision_data = limit_orders[limit_orders['decision_type'] == decision_type]
                if not decision_data.empty:
                    # Calculate percentage difference from current price
                    pct_diff = ((decision_data['price_limit'] - decision_data['price']) / decision_data['price']) * 100
                    plt.scatter(decision_data['current_ratio'],
                              pct_diff,
                              c=colors[decision_type],
                              marker='o',
                              label=f'{decision_type} Limit Price',
                              alpha=0.6)
            
            # Market orders will all be at 0% difference
            market_orders = df[df['order_type'] == 'market']
            for decision_type in ['Buy', 'Sell']:
                decision_data = market_orders[market_orders['decision_type'] == decision_type]
                if not decision_data.empty:
                    plt.scatter(decision_data['current_ratio'],
                              np.zeros(len(decision_data)),  # All at 0% difference
                              c=colors[decision_type],
                              marker='x',
                              label=f'{decision_type} Market Price',
                              alpha=0.6)
            
            ylabel = 'Price Difference from Current Price (%)'
            title = 'Order Price Differences by P/F Ratio'
        
        elif plot_type == 'decision_distribution':
            counts = pd.crosstab(df['current_ratio'], df['decision_type'], normalize='index')
            counts.plot(kind='bar', stacked=True, 
                       color=[colors[col] for col in counts.columns])
            ylabel = 'Proportion of Decisions'
            title = 'Decision Distribution by P/F Ratio'
            
        elif plot_type == 'order_types':
            counts = pd.crosstab(df['current_ratio'], df['order_type'], normalize='index')
            counts.plot(kind='bar', stacked=True, 
                       color=[colors[col] for col in counts.columns])
            ylabel = 'Proportion of Orders'
            title = 'Order Type Distribution by P/F Ratio'
            
        elif plot_type == 'quantities':
            for decision_type, color in colors.items():
                decision_data = df[df['decision_type'] == decision_type]
                if not decision_data.empty:
                    plt.scatter(decision_data['current_ratio'], 
                              decision_data['quantity'],
                              c=color, label=decision_type, alpha=0.6)
            ylabel = 'Quantity'
            title = 'Trade Quantities by P/F Ratio and Decision Type'
        
        plt.title(title)
        plt.xlabel('Price/Fundamental Ratio')
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    def _plot_ratio_patterns(self, df: pd.DataFrame, save_dir: Path):
        """Generate visualizations of patterns across ratios"""
        plt.style.use('default')
        save_dir.mkdir(parents=True, exist_ok=True)
        df['current_ratio'] = df['current_ratio'].round(2)
        
        decision_colors = {'Buy': 'green', 'Sell': 'red', 'Hold': 'gray'}
        order_colors = {'market': 'blue', 'limit': 'orange'}
        
        # Generate standard plots
        plot_types = ['decision_distribution', 'order_types', 'quantities', 'price_analysis']
        for plot_type in plot_types:
            colors = decision_colors if plot_type != 'order_types' else order_colors
            self._create_plot(df, plot_type, colors)
            plt.savefig(save_dir / f'{plot_type}.png', bbox_inches='tight')
            plt.close()
        
        # Save statistics summary
        self._save_statistics_summary(df, save_dir)
    
    def _save_statistics_summary(self, df: pd.DataFrame, save_dir: Path):
        """Save key statistics to a single file"""
        stats = []
        for ratio in sorted(df['current_ratio'].unique()):
            ratio_df = df[df['current_ratio'] == ratio]
            stats.extend([
                f"\nRatio {ratio:.2f}:",
                f"Decision types: {dict(ratio_df['decision_type'].value_counts())}",
                f"Order types: {dict(ratio_df['order_type'].value_counts())}",
                f"Average quantities: {dict(ratio_df.groupby('decision_type')['quantity'].mean())}",
                f"Replace decisions: {dict(ratio_df['replace_decision'].value_counts())}"
            ])
        
        with open(save_dir / "ratio_stats.txt", "w") as f:
            f.write("\n".join(stats))
    
    def generate_ratio_report(self, analysis: Dict) -> str:
        """Generate a human-readable report of the ratio analysis"""
        report = ["=== Price/Fundamental Ratio Analysis Report ===\n"]
        
        # Overall summary
        summary = analysis['summary']
        report.append(f"Total Decisions Analyzed: {summary['total_decisions']}")
        
        # Decision distribution by ratio with detailed reasoning
        report.append("\nDetailed Analysis by Ratio:")
        for ratio in sorted(analysis['decision_distributions'].keys()):
            report.append(f"\n{'='*40}")
            report.append(f"Ratio {ratio:.2f}:")
            
            # Decision distribution
            dist = analysis['decision_distributions'][ratio]
            report.append("\nDecision Distribution:")
            for decision_type, pct in dist.items():
                report.append(f"  {decision_type}: {pct*100:.1f}%")
            
            # Detailed decisions with reasoning
            report.append("\nDetailed Decisions:")
            for decision in analysis['reasoning_summary'][ratio]['decisions']:
                report.append(f"\n  Decision Type: {decision['decision_type']}")
                report.append(f"  Quantity: {decision['quantity']}")
                if decision['order_type']:
                    report.append(f"  Order Type: {decision['order_type']}")
                if decision['price_limit']:
                    report.append(f"  Price Limit: {decision['price_limit']}")
                report.append(f"  Replace Decision: {decision['replace_decision']}")
                report.append(f"  Reasoning:\n    {decision['reasoning']}")
                report.append("-" * 30)
        
        return "\n".join(report)
        

    def analyze_single_decision(self) -> Dict:
        """Analyze a single decision scenario
        
        Returns:
            Dict containing the scenario results including signals, decision, and context
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