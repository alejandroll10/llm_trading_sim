"""Main orchestrator for generating all simulation plots."""

import pandas as pd
from pathlib import Path
from typing import Dict

from utils.csv_loader import load_csv
from visualization.plot_utils import clean_data, save_plot
from visualization.plots import price_plots, agent_plots, trading_plots, decision_plots, valuation_plots, order_plots


class PlotGenerator:
    """Orchestrates the generation of all simulation plots."""

    def __init__(self, simulation):
        """
        Initialize the plot generator.

        Args:
            simulation: BaseSimulation instance with completed run
        """
        self.simulation = simulation
        self.scenario_name = simulation.sim_type.lower()

        # Set up directory paths
        self.dated_plots_dir = simulation.run_dir / 'plots'
        self.dated_plots_dir.mkdir(exist_ok=True)

        scenario_dir = Path('logs') / 'latest_sim' / simulation.sim_type
        scenario_dir.mkdir(parents=True, exist_ok=True)
        self.scenario_plots_dir = scenario_dir / 'plots'
        self.scenario_plots_dir.mkdir(exist_ok=True)

        # Prepare commonly used data
        self.history = simulation.data_recorder.history
        self.data_dir = simulation.run_dir / 'data'

    def save_all_plots(self):
        """Generate and save all plots for the simulation."""
        print("Generating plots...")

        # Generate price plots
        self._generate_price_plots()

        # Generate agent plots
        self._generate_agent_plots()

        # Generate trading flow plots
        self._generate_trading_plots()

        # Generate decision plots
        self._generate_decision_plots()

        # Generate valuation plots
        self._generate_valuation_plots()

        # Generate order flow plots
        self._generate_order_plots()

        print("All plots generated successfully!")

    def _generate_price_plots(self):
        """Generate price-related plots."""
        try:
            print("  Processing price data...")

            rounds = clean_data([h.get('round') for h in self.history])
            fundamental_prices = clean_data([h.get('fundamental_price') for h in self.history])
            last_trade_prices = clean_data([h.get('last_trade_price') for h in self.history])
            midpoint_prices = clean_data([h.get('midpoint') for h in self.history])
            best_bids = clean_data([h.get('best_bid') for h in self.history])
            best_asks = clean_data([h.get('best_ask') for h in self.history])
            short_interest = clean_data([h.get('short_interest') for h in self.history])
            # Get trade counts from trades list in history
            num_trades = [len(h.get('trades', [])) for h in self.history]

            # Price vs Fundamental
            fig = price_plots.plot_price_vs_fundamental(
                rounds, fundamental_prices, last_trade_prices, midpoint_prices, num_trades
            )
            save_plot(fig, 'price_vs_fundamental', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            # Bid-Ask Spread
            fig = price_plots.plot_bid_ask_spread(
                rounds, fundamental_prices, midpoint_prices, best_bids, best_asks
            )
            save_plot(fig, 'bid_ask_spread', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            # Net Short Exposure
            fig = price_plots.plot_net_short_exposure(rounds, short_interest)
            save_plot(fig, 'net_short_exposure', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

        except Exception as e:
            print(f"  Error creating price plots: {str(e)}")

    def _generate_agent_plots(self):
        """Generate agent-related plots."""
        try:
            print("  Processing agent data...")
            agent_data_path = self.data_dir / 'agent_data.csv'

            agent_df = load_csv(agent_data_path, "agent data")
            if agent_df is None:
                return

            # Calculate initial values
            initial_values = self._calculate_initial_values(agent_df)

            # Dividend accumulation
            fig = agent_plots.plot_dividend_accumulation(agent_df)
            if fig:
                save_plot(fig, 'agent_dividend_accumulation', self.scenario_name,
                         self.dated_plots_dir, self.scenario_plots_dir)

            # Wealth composition (final)
            fig = agent_plots.plot_wealth_composition_final(agent_df)
            save_plot(fig, 'wealth_composition', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            # Wealth composition over time (per agent type)
            for agent_type in agent_df['agent_type'].unique():
                try:
                    fig = agent_plots.plot_wealth_composition_overtime(agent_df, agent_type)
                    save_plot(fig, f'wealth_composition_{agent_type.lower()}_overtime',
                             self.scenario_name, self.dated_plots_dir, self.scenario_plots_dir)
                except Exception as e:
                    print(f"  Error creating wealth composition time series for {agent_type}: {str(e)}")

            # Absolute value plots for various metrics
            metrics = [
                ('total_shares', 'Total Share Holdings (Available + Committed)'),
                ('available_shares', 'Available Shares (Not in Orders)'),
                ('committed_shares', 'Committed Shares (Locked in Orders)'),
                ('borrowed_shares', 'Borrowed Shares'),
                ('net_shares', 'Net Share Position (Total - Borrowed)'),
                ('cash', 'Trading Cash Holdings'),
                ('total_value', 'Total Wealth')
            ]

            for metric, title in metrics:
                fig = agent_plots.plot_agent_metric_absolute(agent_df, metric, title)
                if fig:
                    save_plot(fig, f'agent_{metric}_absolute', self.scenario_name,
                             self.dated_plots_dir, self.scenario_plots_dir)

            # Change plots for share metrics
            share_metrics = [
                ('total_shares', 'Change in Total Shares'),
                ('available_shares', 'Change in Available Shares'),
                ('net_shares', 'Change in Net Shares'),
                ('borrowed_shares', 'Change in Borrowed Shares'),
            ]

            for metric, label in share_metrics:
                fig = agent_plots.plot_agent_metric_change(agent_df, metric, label, initial_values)
                if fig:
                    save_plot(fig, f'agent_{metric}_change', self.scenario_name,
                             self.dated_plots_dir, self.scenario_plots_dir)

            # Cash plots
            fig = agent_plots.plot_cash_change(agent_df, initial_values)
            save_plot(fig, 'agent_cash_change', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            fig = agent_plots.plot_cash_returns(agent_df, initial_values)
            save_plot(fig, 'agent_cash_returns', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            # Wealth returns
            fig = agent_plots.plot_wealth_returns(agent_df, initial_values)
            save_plot(fig, 'agent_wealth_returns', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            fig = agent_plots.plot_excess_returns(agent_df, initial_values)
            save_plot(fig, 'agent_excess_returns', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            # Leverage plots (only if leverage is being used)
            if 'borrowed_cash' in agent_df.columns and agent_df['borrowed_cash'].sum() > 0:
                print("  Processing leverage metrics...")

                # Get leverage parameters from simulation
                leverage_params = self.simulation.params.get('AGENT_PARAMS', {}).get('leverage_params', {})
                maintenance_margin = leverage_params.get('maintenance_margin', 0.25)
                initial_margin = leverage_params.get('initial_margin', 0.5)

                # Borrowed cash plot
                fig = agent_plots.plot_borrowed_cash(agent_df)
                if fig:
                    save_plot(fig, 'leverage_borrowed_cash', self.scenario_name,
                             self.dated_plots_dir, self.scenario_plots_dir)

                # Margin ratios plot
                fig = agent_plots.plot_margin_ratios(agent_df, maintenance_margin, initial_margin)
                if fig:
                    save_plot(fig, 'leverage_margin_ratios', self.scenario_name,
                             self.dated_plots_dir, self.scenario_plots_dir)

                # Leverage interest plot
                fig = agent_plots.plot_leverage_interest(agent_df)
                if fig:
                    save_plot(fig, 'leverage_interest_paid', self.scenario_name,
                             self.dated_plots_dir, self.scenario_plots_dir)

                # Leverage heatmap
                fig = agent_plots.plot_leverage_heatmap(agent_df)
                if fig:
                    save_plot(fig, 'leverage_usage_heatmap', self.scenario_name,
                             self.dated_plots_dir, self.scenario_plots_dir)

        except Exception as e:
            print(f"  Error processing agent data: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _generate_trading_plots(self):
        """Generate trading flow plots."""
        try:
            trade_data_path = self.data_dir / 'trade_data.csv'
            trade_df = load_csv(trade_data_path, "trade data", silent=True)
            if trade_df is None:
                return

            # Load agent data for type mapping
            agent_data_path = self.data_dir / 'agent_data.csv'
            agent_df = load_csv(agent_data_path, "agent data (for type mapping)")
            if agent_df is None:
                return

            agent_type_map = agent_df.groupby('agent_id')['agent_type'].first().to_dict()

            # Trading flow
            fig = trading_plots.plot_trading_flow(trade_df, agent_type_map)
            save_plot(fig, 'trading_flow', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            # Cumulative trading flow
            fig = trading_plots.plot_cumulative_trading_flow(trade_df, agent_type_map)
            save_plot(fig, 'cumulative_trading_flow', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

        except Exception as e:
            print(f"  Error creating trading flow plots: {str(e)}")

    def _generate_decision_plots(self):
        """Generate decision analysis plots."""
        try:
            decisions_path = self.simulation.run_dir / 'structured_decisions.csv'
            decisions_df = load_csv(decisions_path, "decision data", silent=True)
            if decisions_df is None:
                return

            # Decision heatmap
            fig = decision_plots.plot_decision_heatmap(decisions_df)
            save_plot(fig, 'decision_heatmap', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            # Decision quantities
            fig = decision_plots.plot_decision_quantities(decisions_df)
            save_plot(fig, 'decision_quantities', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            # Word clouds
            wordcloud_figs = decision_plots.generate_all_wordclouds(decisions_df)
            for key, fig in wordcloud_figs.items():
                if key == 'all':
                    save_plot(fig, 'reasoning_wordcloud_all', self.scenario_name,
                             self.dated_plots_dir, self.scenario_plots_dir)
                else:
                    save_plot(fig, f'reasoning_wordcloud_{key.lower()}', self.scenario_name,
                             self.dated_plots_dir, self.scenario_plots_dir)

        except Exception as e:
            print(f"  Error creating decision plots: {str(e)}")

    def _generate_valuation_plots(self):
        """Generate valuation analysis plots."""
        try:
            decisions_path = self.simulation.run_dir / 'structured_decisions.csv'
            decisions_df = load_csv(decisions_path, "decision data", silent=True)
            if decisions_df is None:
                return

            # Skip if no valuation data
            if 'valuation' not in decisions_df.columns or decisions_df['valuation'].isna().all():
                return

            print("  Processing valuation data...")

            # Prepare price data
            rounds = clean_data([h.get('round') for h in self.history])
            price_data = clean_data([h.get('price') for h in self.history])
            fundamental_data = clean_data([h.get('fundamental_price') for h in self.history])
            # Get trade counts from trades list in history
            num_trades = [len(h.get('trades', [])) for h in self.history]

            # Agent valuations
            fig = valuation_plots.plot_agent_valuations(
                decisions_df, self.history, rounds, price_data, fundamental_data, num_trades
            )
            if fig:
                save_plot(fig, 'agent_valuations', self.scenario_name,
                         self.dated_plots_dir, self.scenario_plots_dir)

            # Valuation dispersion
            fig = valuation_plots.plot_valuation_dispersion(decisions_df, self.history)
            if fig:
                save_plot(fig, 'valuation_dispersion', self.scenario_name,
                         self.dated_plots_dir, self.scenario_plots_dir)

            # Price expectations
            fig = valuation_plots.plot_price_prediction_accuracy(decisions_df, price_data, num_trades)
            if fig:
                save_plot(fig, 'price_prediction_accuracy', self.scenario_name,
                         self.dated_plots_dir, self.scenario_plots_dir)

            # Price prediction errors
            fig = valuation_plots.plot_price_prediction_errors(decisions_df, price_data)
            if fig:
                save_plot(fig, 'price_prediction_errors', self.scenario_name,
                         self.dated_plots_dir, self.scenario_plots_dir)

            # Combined valuation vs expectations plot
            fig = valuation_plots.plot_valuation_vs_expectations(
                decisions_df, self.history, rounds, price_data, fundamental_data, num_trades
            )
            if fig:
                save_plot(fig, 'valuation_vs_expectations', self.scenario_name,
                         self.dated_plots_dir, self.scenario_plots_dir)

        except Exception as e:
            print(f"  Error creating valuation plots: {str(e)}")

    def _generate_order_plots(self):
        """Generate order flow plots."""
        try:
            order_data_path = self.data_dir / 'order_data.csv'
            order_df = load_csv(order_data_path, "order data", silent=True)
            if order_df is None:
                return

            # Load agent data for type mapping
            agent_data_path = self.data_dir / 'agent_data.csv'
            agent_df = load_csv(agent_data_path, "agent data (for type mapping)")
            if agent_df is None:
                return

            agent_type_map = agent_df.groupby('agent_id')['agent_type'].first().to_dict()

            # Order flow by type
            fig = order_plots.plot_order_flow_by_type(order_df, agent_type_map)
            save_plot(fig, 'order_flow_by_type', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            # Net order flow
            fig = order_plots.plot_order_flow_net(order_df, agent_type_map)
            save_plot(fig, 'order_flow_net', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

            # Aggregated order flow
            fig = order_plots.plot_order_flow_aggregated(order_df, agent_type_map)
            save_plot(fig, 'order_flow_aggregated', self.scenario_name,
                     self.dated_plots_dir, self.scenario_plots_dir)

        except Exception as e:
            print(f"  Error creating order flow plots: {str(e)}")

    def _calculate_initial_values(self, agent_df: pd.DataFrame) -> Dict:
        """
        Calculate initial values for each agent type.

        Args:
            agent_df: DataFrame with agent data

        Returns:
            Dict of initial values by agent type
        """
        initial_values = {}

        for agent_type in agent_df['agent_type'].unique():
            # Find the minimum round for this agent type
            min_round = agent_df[agent_df['agent_type'] == agent_type]['round'].min()
            # Get data from the first round
            type_data = agent_df[(agent_df['agent_type'] == agent_type) & (agent_df['round'] == min_round)]

            if not type_data.empty:
                initial_values[agent_type] = {
                    'total_shares': type_data['total_shares'].sum(),
                    'borrowed_shares': type_data['borrowed_shares'].sum(),
                    'net_shares': type_data['net_shares'].sum(),
                    'cash': type_data['cash'].sum(),
                    'total_value': type_data['total_value'].sum()
                }
                print(f"  Initial values for {agent_type}: Cash=${initial_values[agent_type]['cash']:.2f}, "
                      f"Total Shares={initial_values[agent_type]['total_shares']}, "
                      f"Value=${initial_values[agent_type]['total_value']:.2f}")

        return initial_values
