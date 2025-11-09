import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import random
from base_sim import BaseSimulation
import warnings
from pathlib import Path
from datetime import datetime
from services.logging_service import LoggingService
from scenarios import get_scenario, list_scenarios
import pandas as pd
import shutil
from wordcloud import WordCloud

def create_run_directory(sim_type: str, description: str = "", parameters: dict = None) -> Path:
    """Create a directory structure that includes simulation type and date"""
    base_dir = Path('logs')
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create nested structure: logs/sim_type/YYYY-MM-DD_HH-MM-SS/
    run_dir = base_dir / sim_type / date_str
    
    # Add description to metadata rather than folder name to keep paths clean
    metadata = {
        'sim_type': sim_type,
        'description': description,
        'timestamp': date_str,
        'parameters': parameters
    }
    
    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    with open(run_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return run_dir

def save_parameters(run_dir: Path, params: dict):
    """Save parameters to a JSON file in both run directory and latest_sim"""
    # Save to dated run directory
    with open(run_dir / 'parameters.json', 'w') as f:
        json.dump(params, f, indent=4)
    
    # Save to latest_sim directory with scenario subfolder
    sim_type = run_dir.parts[-2]  # Extract scenario name from run_dir path
    latest_dir = Path('logs') / 'latest_sim'
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    # Create scenario directory if it doesn't exist
    scenario_dir = latest_dir / sim_type
    scenario_dir.mkdir(exist_ok=True)
    
    # Save parameters to scenario directory
    with open(scenario_dir / 'parameters.json', 'w') as f:
        json.dump(params, f, indent=4)

def save_plots(simulation, params: dict):
    """Save all simulation plots to both run directory and latest_sim"""
    print("Generating plots...")  # Add logging for visibility
    
    # Create plots directory in dated run_dir
    dated_plots_dir = simulation.run_dir / 'plots'
    dated_plots_dir.mkdir(exist_ok=True)
    
    # Create scenario-specific plots directory in latest_sim
    scenario_name = simulation.sim_type.lower()
    latest_sim_dir = Path('logs') / 'latest_sim'
    latest_sim_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir = latest_sim_dir / simulation.sim_type
    scenario_dir.mkdir(exist_ok=True)
    scenario_plots_dir = scenario_dir / 'plots'
    scenario_plots_dir.mkdir(exist_ok=True)

    # Helper function to save with scenario name suffix
    def save_plot_with_suffix(base_name):
        try:
            plt.savefig(dated_plots_dir / f'{base_name}_{scenario_name}.png')
            plt.savefig(scenario_plots_dir / f'{base_name}_{scenario_name}.png')
            plt.close()
            print(f"  Saved plot: {base_name}")  # Add logging for each plot
        except Exception as e:
            print(f"  Error saving plot {base_name}: {str(e)}")

    # Helper function to ensure data is numeric and finite
    def clean_data(data_list):
        if data_list is None:
            return []
        cleaned = []
        for x in data_list:
            try:
                if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                    cleaned.append(np.nan)
                else:
                    cleaned.append(float(x))
            except (ValueError, TypeError):
                cleaned.append(np.nan)
        return cleaned

    try:
        # Get history and extract price data
        print("  Processing price data...")
        history = simulation.data_recorder.history
        rounds = clean_data([h.get('round') for h in history])
        prices = clean_data([h.get('price') for h in history])
        fundamental_prices = clean_data([h.get('fundamental_price') for h in history])
        last_trade_prices = clean_data([h.get('last_trade_price') for h in history])
        midpoint_prices = clean_data([h.get('midpoint') for h in history])
        best_bids = clean_data([h.get('best_bid') for h in history])
        best_asks = clean_data([h.get('best_ask') for h in history])

        # Price vs Fundamental Value plot
        plt.figure(figsize=(12, 6))
        plt.plot(rounds, fundamental_prices, label='Fundamental Value', linestyle='--', linewidth=2, color='green')
        plt.plot(rounds, last_trade_prices, label='Last Trade', color='red', alpha=0.8, linewidth=2)
        plt.plot(rounds, midpoint_prices, label='Midpoint', color='purple', alpha=0.8, linewidth=2)
        
        plt.xlabel('Round')
        plt.ylabel('Price')
        plt.title('Market Price Evolution')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        save_plot_with_suffix('price_vs_fundamental')

        # Add bid-ask spread plot with fundamental value
        plt.figure(figsize=(12, 6))
        
        # Plot midpoint as main line
        plt.plot(rounds, midpoint_prices, label='Midpoint Price', color='blue', linewidth=2)
        
        # Add fundamental value line
        plt.plot(rounds, fundamental_prices, label='Fundamental Value', linestyle='--', linewidth=2, color='green')
        
        # Plot bid and ask as filled area around midpoint
        plt.fill_between(rounds, best_bids, best_asks, 
                        alpha=0.2, color='gray', label='Bid-Ask Spread')
        
        # Add actual bid and ask lines for clarity
        plt.plot(rounds, best_bids, '--', color='red', alpha=0.6, label='Best Bid')
        plt.plot(rounds, best_asks, '--', color='green', alpha=0.6, label='Best Ask')
        
        plt.xlabel('Round')
        plt.ylabel('Price')
        plt.title('Price and Bid-Ask Spread')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_plot_with_suffix('bid_ask_spread')

        # Net short exposure over time
        short_interest = clean_data([h.get('short_interest') for h in history])
        plt.figure(figsize=(12, 6))
        plt.plot(rounds, [-s for s in short_interest], label='Net Short Exposure', color='blue', linewidth=2)
        plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        plt.xlabel('Round')
        plt.ylabel('Shares')
        plt.title('Net Short Exposure Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_plot_with_suffix('net_short_exposure')
    except Exception as e:
        print(f"  Error creating price plots: {str(e)}")

    # Add new agent-specific plots with return calculations
    try:
        print("  Processing agent data...")
        agent_data_path = simulation.run_dir / 'data' / 'agent_data.csv'
        if not agent_data_path.exists():
            print(f"  Agent data file not found: {agent_data_path}")
            return
            
        agent_df = pd.read_csv(agent_data_path)
        if agent_df.empty:
            print("  Agent data file is empty")
            return
        
        # Create initial values dictionary by agent_type
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
                print(f"  Initial values for {agent_type}: Cash=${initial_values[agent_type]['cash']:.2f}, " +
                      f"Total Shares={initial_values[agent_type]['total_shares']}, Value=${initial_values[agent_type]['total_value']:.2f}")
        
        # Add plot for dividend cash accumulation
        try:
            if 'dividend_cash' in agent_df.columns:
                plt.figure(figsize=(12, 6))
                
                # Group by round and agent_type, sum for dividend_cash
                grouped = agent_df.groupby(['round', 'agent_type'])['dividend_cash'].sum().unstack()
                
                # Plot dividend accumulation
                grouped.plot(kind='line', marker='o')
                
                plt.xlabel('Round')
                plt.ylabel('Accumulated Dividends & Interest ($)')
                plt.title('Dividend & Interest Accumulation by Agent Type')
                plt.legend(title='Agent Type')
                plt.grid(True, alpha=0.3)
                
                save_plot_with_suffix('agent_dividend_accumulation')
        except Exception as e:
            print(f"  Error creating dividend accumulation plot: {str(e)}")
        
        # Add wealth composition plot - show breakdown of wealth components
        try:
            plt.figure(figsize=(12, 8))
            
            # Pick a sample round near the end for final composition
            final_round = agent_df['round'].max()
            final_data = agent_df[agent_df['round'] == final_round]
            
            # Aggregate by agent type
            wealth_components = final_data.groupby('agent_type').agg({
                'cash': 'sum',
                'dividend_cash': 'sum' if 'dividend_cash' in agent_df.columns else (lambda x: 0),
                'share_value': 'sum',
                'total_value': 'sum'
            })
            
            # Plot as stacked bar chart
            ax = wealth_components[['cash', 'dividend_cash' if 'dividend_cash' in agent_df.columns else 'cash', 'share_value']].plot(
                kind='bar', 
                stacked=True,
                figsize=(12, 6),
                color=['#3498db', '#2ecc71', '#e74c3c']
            )
            
            # Add total as text on top of bars
            for i, total in enumerate(wealth_components['total_value']):
                ax.text(i, total + 5, f'${total:.0f}', ha='center', fontweight='bold')
            
            plt.xlabel('Agent Type')
            plt.ylabel('Wealth Value ($)')
            plt.title(f'Wealth Composition by Agent Type (Round {final_round})')
            plt.legend(['Trading Cash', 'Dividend & Interest Cash', 'Share Value'])
            plt.grid(True, alpha=0.3, axis='y')
            
            save_plot_with_suffix('wealth_composition')
            
            # Add a time series showing wealth composition for each agent type
            for agent_type in agent_df['agent_type'].unique():
                try:
                    plt.figure(figsize=(12, 6))
                    
                    # Filter for this agent type only
                    agent_type_data = agent_df[agent_df['agent_type'] == agent_type]
                    
                    # Group by round
                    by_round = agent_type_data.groupby('round').agg({
                        'cash': 'sum',
                        'dividend_cash': 'sum' if 'dividend_cash' in agent_df.columns else (lambda x: 0),
                        'share_value': 'sum'
                    })
                    
                    # Create stacked area plot
                    by_round.plot(
                        kind='area',
                        stacked=True,
                        alpha=0.7,
                        color=['#3498db', '#2ecc71', '#e74c3c']
                    )
                    
                    plt.xlabel('Round')
                    plt.ylabel('Value ($)')
                    plt.title(f'Wealth Composition Over Time: {agent_type}')
                    plt.legend(['Trading Cash', 'Dividend & Interest Cash', 'Share Value'])
                    plt.grid(True, alpha=0.3)
                    
                    save_plot_with_suffix(f'wealth_composition_{agent_type.lower()}_overtime')
                except Exception as e:
                    print(f"  Error creating wealth composition time series for {agent_type}: {str(e)}")
                
        except Exception as e:
            print(f"  Error creating wealth composition plot: {str(e)}")
        
        # 1. Absolute value plots - Show raw values for all metrics
        for metric, title in [
            ('total_shares', 'Total Share Holdings (Available + Committed)'),
            ('available_shares', 'Available Shares (Not in Orders)'),
            ('committed_shares', 'Committed Shares (Locked in Orders)'),
            ('borrowed_shares', 'Borrowed Shares'),
            ('net_shares', 'Net Share Position (Total - Borrowed)'),
            ('cash', 'Trading Cash Holdings'),
            ('total_value', 'Total Wealth')
        ]:
            try:
                if metric not in agent_df.columns:
                    continue

                plt.figure(figsize=(12, 6))

                # Group by round and agent_type, sum for the metric
                grouped = agent_df.groupby(['round', 'agent_type'])[metric].sum().unstack()

                # Plot absolute values
                grouped.plot(kind='line', marker='o')

                plt.xlabel('Round')
                plt.ylabel(f'{title}')
                plt.title(f'{title} by Agent Type')
                plt.legend(title='Agent Type')
                plt.grid(True, alpha=0.3)

                save_plot_with_suffix(f'agent_{metric}_absolute')
            except Exception as e:
                print(f"  Error creating {metric} absolute plot: {str(e)}")
        
        # 2. For share-related metrics - Show absolute change
        for metric, label in [
            ('total_shares', 'Change in Total Shares'),
            ('available_shares', 'Change in Available Shares'),
            ('net_shares', 'Change in Net Shares'),
            ('borrowed_shares', 'Change in Borrowed Shares'),
        ]:
            try:
                if metric not in agent_df.columns:
                    continue

                plt.figure(figsize=(12, 6))

                # Group by round and agent_type, sum metric
                grouped = agent_df.groupby(['round', 'agent_type'])[metric].sum().unstack()

                # Calculate absolute change from initial values
                changes = pd.DataFrame(index=grouped.index, columns=grouped.columns)

                for agent_type in grouped.columns:
                    if agent_type in initial_values:
                        initial = initial_values[agent_type].get(metric, 0)
                        changes[agent_type] = grouped[agent_type] - initial

                # Plot changes
                changes.plot(kind='line', marker='o')

                plt.xlabel('Round')
                plt.ylabel(label)
                plt.title(f'{label} by Agent Type')
                plt.legend(title='Agent Type')
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

                save_plot_with_suffix(f'agent_{metric}_change')
            except Exception as e:
                print(f"  Error creating {metric} change plot: {str(e)}")
            
        # 3. For cash - Show both percentage return and absolute change
        try:
            # 3.1 Absolute change in cash
            plt.figure(figsize=(12, 6))
            
            # Group by round and agent_type, sum cash
            grouped = agent_df.groupby(['round', 'agent_type'])['cash'].sum().unstack()
            
            # Calculate absolute change from initial cash
            cash_changes = pd.DataFrame(index=grouped.index, columns=grouped.columns)
            
            for agent_type in grouped.columns:
                if agent_type in initial_values:
                    initial = initial_values[agent_type]['cash']
                    cash_changes[agent_type] = grouped[agent_type] - initial
            
            # Plot cash changes
            cash_changes.plot(kind='line', marker='o')
            
            plt.xlabel('Round')
            plt.ylabel('Change in Trading Cash ($)')
            plt.title('Change in Trading Cash Holdings by Agent Type')
            plt.legend(title='Agent Type')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            save_plot_with_suffix('agent_cash_change')
            
            # 3.2 Percentage return on cash
            plt.figure(figsize=(12, 6))
            
            # Calculate percentage return on cash
            cash_returns = pd.DataFrame(index=grouped.index, columns=grouped.columns)
            
            for agent_type in grouped.columns:
                if agent_type in initial_values:
                    initial = initial_values[agent_type]['cash']
                    if initial > 0:
                        cash_returns[agent_type] = (grouped[agent_type] / initial - 1) * 100
            
            # Plot cash returns
            cash_returns.plot(kind='line', marker='o')
            
            plt.xlabel('Round')
            plt.ylabel('Change in Trading Cash (%)')
            plt.title('Percentage Change in Trading Cash by Agent Type')
            plt.legend(title='Agent Type')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            save_plot_with_suffix('agent_cash_returns')
        except Exception as e:
            print(f"  Error creating cash plots: {str(e)}")
            
        # 4. For total value - Show percentage return (most meaningful)
        try:
            plt.figure(figsize=(12, 6))
            
            # Group by round and agent_type, sum total value
            grouped = agent_df.groupby(['round', 'agent_type'])['total_value'].sum().unstack()
            
            # Calculate percentage return on total value
            value_returns = pd.DataFrame(index=grouped.index, columns=grouped.columns)
            
            # Also calculate excess returns over risk-free rate
            excess_returns = pd.DataFrame(index=grouped.index, columns=grouped.columns)
            
            # Risk-free rate (per round)
            per_round_rf = 0.001
            
            for agent_type in grouped.columns:
                if agent_type in initial_values:
                    initial = initial_values[agent_type]['total_value']
                    if initial > 0:
                        # Raw returns
                        value_returns[agent_type] = (grouped[agent_type] / initial - 1) * 100
                        
                        # Excess returns (subtracting compounded risk-free rate)
                        for idx, round_num in enumerate(value_returns.index):
                            # Account for round numbering (round starts at 0 or 1?)
                            # Adjust t based on your round numbering convention
                            t = round_num - min(value_returns.index) if min(value_returns.index) > 0 else round_num
                            
                            # Compound risk-free rate: (1+rf)^t - 1
                            rf_return = ((1 + per_round_rf) ** t - 1) * 100
                            
                            # Excess return = actual return - risk-free return
                            if not pd.isna(value_returns.at[round_num, agent_type]):
                                excess_returns.at[round_num, agent_type] = value_returns.at[round_num, agent_type] - rf_return
            
            # Plot total value returns
            value_returns.plot(kind='line', marker='o')
            
            plt.xlabel('Round')
            plt.ylabel('Change in Total Wealth (%)')
            plt.title('Percentage Change on Total Wealth by Agent Type')
            plt.legend(title='Agent Type')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            save_plot_with_suffix('agent_wealth_returns')
            
            # Plot excess returns (over risk-free rate)
            plt.figure(figsize=(12, 6))
            excess_returns.plot(kind='line', marker='o')
            
            plt.xlabel('Round')
            plt.ylabel('Excess Return (%)')
            plt.title('Excess Returns Over Risk-Free Rate (5% Per Round)')
            plt.legend(title='Agent Type')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add a line showing cumulative risk-free return for reference
            rf_line = [(((1 + per_round_rf) ** t - 1) * 100) for t in range(len(grouped.index))]
            plt.plot(grouped.index, [0] * len(grouped.index), 'k--', alpha=0.5, label='Risk-Free Baseline')
            
            save_plot_with_suffix('agent_excess_returns')
        except Exception as e:
            print(f"  Error creating wealth returns plot: {str(e)}")
            
    except Exception as e:
        print(f"  Error processing agent data: {str(e)}")
        import traceback
        print(traceback.format_exc())

    # Add trading flow analysis
    trade_data_path = simulation.run_dir / 'data' / 'trade_data.csv'
    trade_df = None
    if trade_data_path.exists():
        try:
            trade_df = pd.read_csv(trade_data_path)
            if trade_df.empty:
                print("  Trade data file exists but is empty")
                trade_df = None
        except pd.errors.EmptyDataError:
            print("  Trade data file exists but has no columns")
            trade_df = None
    
    if trade_df is not None and not trade_df.empty:
        try:
            agent_df = pd.read_csv(agent_data_path)
            
            # Create agent_id to agent_type mapping
            agent_type_map = agent_df.groupby('agent_id')['agent_type'].first().to_dict()
            
            # Add buyer and seller types to trade data
            trade_df['buyer_type'] = trade_df['buyer_id'].map(agent_type_map)
            trade_df['seller_type'] = trade_df['seller_id'].map(agent_type_map)
            
            # Group by round and calculate volume between agent types
            trade_volume = trade_df.groupby(['round', 'buyer_type', 'seller_type'])['quantity'].sum().reset_index()
    
            # Get unique agent types
            agent_types = sorted(set(agent_type_map.values()))
            
            for buyer_type in agent_types:
                for seller_type in agent_types:
                    mask = (trade_volume['buyer_type'] == buyer_type) & (trade_volume['seller_type'] == seller_type)
                    if mask.any():  # Only plot if there are trades between these types
                        plt.plot(trade_volume[mask]['round'], 
                                trade_volume[mask]['quantity'],
                                label=f'{seller_type} → {buyer_type}',
                                linewidth=2)
            
            plt.xlabel('Round')
            plt.ylabel('Trading Volume')
            plt.title('Trading Volume Between Agent Types')
            plt.legend(title='Trade Direction')
            plt.grid(True, alpha=0.3)
            
            save_plot_with_suffix('trading_flow')

            # Add cumulative net flow plot
            plt.figure(figsize=(12, 6))
            
            for agent_type in agent_types:
                # Calculate net flow (positive when buying, negative when selling)
                buying_mask = trade_df['buyer_type'] == agent_type
                selling_mask = trade_df['seller_type'] == agent_type
                
                trade_df['net_flow'] = 0
                trade_df.loc[buying_mask, 'net_flow'] = trade_df.loc[buying_mask, 'quantity']
                trade_df.loc[selling_mask, 'net_flow'] = -trade_df.loc[selling_mask, 'quantity']
                
                # Calculate cumulative net flow for this agent type
                net_flow = trade_df.groupby('round')['net_flow'].sum()
                cumulative_flow = net_flow.cumsum()
                
                plt.plot(cumulative_flow.index, cumulative_flow.values,
                        label=agent_type,
                        linewidth=2)
            
            plt.xlabel('Round')
            plt.ylabel('Cumulative Net Trading Volume')
            plt.title('Cumulative Net Trading Flow by Agent Type')
            plt.legend(title='Agent Type')
            plt.grid(True, alpha=0.3)
            
            save_plot_with_suffix('cumulative_trading_flow')
        except Exception as e:
            print(f"Error creating trading flow plots: {str(e)}")

    # Add decision analysis plots
    decisions_path = simulation.run_dir / 'structured_decisions.csv'
    if decisions_path.exists():
        decisions_df = pd.read_csv(decisions_path)
        
        # Convert decision types to binary (Buy = 1, Sell = -1)
        decisions_df['decision_value'] = decisions_df['decision'].map({'Buy': 1, 'Sell': -1})
        
        # Plot decision heatmap by agent type
        plt.figure(figsize=(12, 6))
        
        # Group by round and agent_type, calculate mean decision (-1 to 1)
        decision_heat = decisions_df.groupby(['round', 'agent_type'])['decision_value'].mean().unstack()
        
        # Plot heatmap
        plt.imshow(decision_heat.T, aspect='auto', cmap='RdYlGn', 
                  vmin=-1, vmax=1, interpolation='nearest')
        
        plt.colorbar(label='Buy (1) vs Sell (-1)')
        plt.xlabel('Round')
        plt.ylabel('Agent Type')
        plt.title('Agent Decision Patterns Over Time')
        
        # Set y-axis labels
        plt.yticks(range(len(decision_heat.columns)), decision_heat.columns)
        
        save_plot_with_suffix('decision_heatmap')

        # Plot decision quantities by agent type
        plt.figure(figsize=(12, 6))
        
        for agent_type in decisions_df['agent_type'].unique():
            agent_mask = decisions_df['agent_type'] == agent_type
            buys = decisions_df[agent_mask & (decisions_df['decision'] == 'Buy')]
            sells = decisions_df[agent_mask & (decisions_df['decision'] == 'Sell')]
            
            # Plot buys and sells
            if not buys.empty:
                plt.scatter(buys['round'], buys['quantity'], 
                          marker='^', label=f'{agent_type} Buys')
            if not sells.empty:
                plt.scatter(sells['round'], -sells['quantity'], 
                          marker='v', label=f'{agent_type} Sells')
        
        plt.xlabel('Round')
        plt.ylabel('Quantity (negative for sells)')
        plt.title('Agent Decision Quantities Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_plot_with_suffix('decision_quantities')

        # Plot reasoning word clouds by agent type
        if 'reasoning' in decisions_df.columns:
            # Create word cloud for each agent type
            for agent_type in decisions_df['agent_type'].unique():
                plt.figure(figsize=(12, 8))
                
                # Get reasoning text for this agent type only
                agent_text = ' '.join(
                    decisions_df[decisions_df['agent_type'] == agent_type]['reasoning']
                    .dropna()
                    .astype(str)
                )
                
                if agent_text.strip():  # Only create word cloud if there's text
                    # Generate and plot word cloud
                    wordcloud = WordCloud(
                        width=1200, 
                        height=800,
                        background_color='white',
                        min_font_size=10
                    ).generate(agent_text)
                    
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'Common Terms in {agent_type} Agent Reasoning')
                    
                    save_plot_with_suffix(f'reasoning_wordcloud_{agent_type.lower()}')
            
            # Also keep the overall word cloud
            all_text = ' '.join(decisions_df['reasoning'].dropna().astype(str))
            if all_text.strip():
                plt.figure(figsize=(12, 8))
                wordcloud = WordCloud(
                    width=1200,
                    height=800,
                    background_color='white',
                    min_font_size=10
                ).generate(all_text)

                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Common Terms in All Agent Reasoning')

                save_plot_with_suffix('reasoning_wordcloud_all')

    # Add valuation analysis plots
    decisions_path = simulation.run_dir / 'structured_decisions.csv'
    if decisions_path.exists():
        try:
            decisions_df = pd.read_csv(decisions_path)
            
            # Skip if no valuation data
            if 'valuation' in decisions_df.columns and not decisions_df['valuation'].isna().all():
                print("  Processing valuation data...")
                
                # 1. Plot agent valuations compared to market price over time
                plt.figure(figsize=(12, 6))
                
                # Add actual market price
                price_data = clean_data([h.get('price') for h in history])
                fundamental_data = clean_data([h.get('fundamental_price') for h in history])
                plt.plot(rounds, price_data, label='Market Price', color='black', linewidth=2)
                plt.plot(rounds, fundamental_data, label='Fundamental Value', color='green', linestyle='--', linewidth=2)
                
                # Group valuations by agent type and round
                agent_valuations = decisions_df.groupby(['round', 'agent_type'])['valuation'].mean().unstack()
                
                # Plot each agent type's valuation
                for agent_type in agent_valuations.columns:
                    plt.plot(agent_valuations.index, agent_valuations[agent_type], 
                           label=f'{agent_type} Valuation', linewidth=1.5, alpha=0.7)
                
                plt.xlabel('Round')
                plt.ylabel('Price / Valuation')
                plt.title('Agent Valuations vs Market Price')
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                
                save_plot_with_suffix('agent_valuations')
                
                # 2. Plot valuation dispersion (box plot)
                plt.figure(figsize=(12, 6))
                
                # Create list of rounds and corresponding valuations for each agent type
                unique_rounds = sorted(decisions_df['round'].unique())
                sampled_rounds = unique_rounds[::max(1, len(unique_rounds)//10)]  # Sample ~10 rounds
                
                # Create a figure with adequate spacing for labels
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Create a list to store handles for legend
                legend_handles = []
                legend_labels = []
                
                for i, round_num in enumerate(sampled_rounds):
                    round_data = decisions_df[decisions_df['round'] == round_num]
                    if not round_data.empty:
                        data = []
                        labels = []
                        for agent_type in round_data['agent_type'].unique():
                            agent_data = round_data[round_data['agent_type'] == agent_type]['valuation']
                            if not agent_data.empty:
                                data.append(agent_data)
                                labels.append(agent_type)
                        
                        # Create a boxplot showing the distribution of valuations
                        boxplot_positions = [i + j*0.8/len(labels) for j in range(len(labels))]
                        bp = ax.boxplot(data, positions=boxplot_positions, widths=0.1, 
                                        patch_artist=True)
                        
                        # Color the boxes according to agent type (using a consistent color scheme)
                        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
                        for j, (patch, color, label) in enumerate(zip(bp['boxes'], colors, labels)):
                            patch.set_facecolor(color)
                            
                            # Only add to legend if this agent type hasn't been added yet
                            if label not in legend_labels:
                                legend_handles.append(patch)
                                legend_labels.append(label)
                        
                        # Add price and fundamental as horizontal lines for this round
                        if i == 0:
                            round_idx = min(round_num, len(history)-1)
                            round_price = history[round_idx].get('price', None)
                            round_fundamental = history[round_idx].get('fundamental_price', None)
                            
                            if round_price is not None:
                                price_line = ax.axhline(y=round_price, color='red', linestyle='-')
                                legend_handles.append(price_line)
                                legend_labels.append('Market Price')
                            
                            if round_fundamental is not None:
                                fund_line = ax.axhline(y=round_fundamental, color='green', linestyle='--')
                                legend_handles.append(fund_line)
                                legend_labels.append('Fundamental Value')
                
                ax.set_xlabel('Round Number')
                ax.set_ylabel('Valuation')
                ax.set_title('Distribution of Agent Valuations by Type')
                
                # Set x-tick positions at the center of each round's boxplots
                xtick_positions = [i + 0.4 for i in range(len(sampled_rounds))]
                ax.set_xticks(xtick_positions)
                ax.set_xticklabels(sampled_rounds)
                
                # Add a legend with all items
                ax.legend(legend_handles, legend_labels, loc='best')
                ax.grid(True, alpha=0.3)
                
                # Adjust layout to make room for labels
                plt.tight_layout()
                
                save_plot_with_suffix('valuation_dispersion')
                
                # 3. Plot price target accuracy
                plt.figure(figsize=(14, 8))  # Larger figure size
                
                # Group by round and agent_type
                price_targets = decisions_df.groupby(['round', 'agent_type'])['price_target'].mean().unstack()
                
                # Calculate next round actual prices
                actual_next_prices = []
                for r in price_targets.index:
                    if r + 1 < len(price_data):
                        actual_next_prices.append(price_data[r + 1])
                    else:
                        actual_next_prices.append(None)
                
                # Plot each agent type's price target
                for agent_type in price_targets.columns:
                    plt.plot(price_targets.index, price_targets[agent_type], 
                           label=f'{agent_type} Target', linewidth=1.5, alpha=0.7)
                
                # Plot actual next round prices
                plt.plot(price_targets.index, actual_next_prices, 
                       label='Actual Next Price', color='black', linewidth=2)
                
                plt.xlabel('Round')
                plt.ylabel('Price')
                plt.title('Agent Price Targets vs Actual Next Prices')
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                
                # Make sure there's enough room for labels
                plt.tight_layout()
                
                save_plot_with_suffix('price_target_accuracy')
                    
                # 4. Plot price target error by agent type
                plt.figure(figsize=(14, 8))  # Larger figure size
                
                error_data = []
                agent_types = []
                
                for agent_type in price_targets.columns:
                    # Calculate errors where we have both targets and actuals
                    errors = []
                    for r in price_targets.index:
                        if r < len(actual_next_prices) and actual_next_prices[r] is not None:
                            target = price_targets.loc[r, agent_type]
                            actual = actual_next_prices[r]
                            if not np.isnan(target) and not np.isnan(actual):
                                errors.append(abs(target - actual) / actual * 100)  # Percent error
                    
                    if errors:
                        error_data.append(errors)
                        agent_types.append(agent_type)
                
                if error_data:
                    plt.boxplot(error_data, labels=agent_types)
                    plt.ylabel('Absolute Percent Error (%)')
                    plt.title('Price Target Accuracy by Agent Type')
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    
                    save_plot_with_suffix('price_target_errors')
        except Exception as e:
            print(f"  Error creating valuation plots: {str(e)}")

    # Plot order analysis plots
    order_data_path = simulation.run_dir / 'data' / 'order_data.csv'
    if order_data_path.exists():
        try:
            order_df = pd.read_csv(order_data_path)
            
            # Add agent types to orders
            agent_type_map = agent_df.groupby('agent_id')['agent_type'].first().to_dict()
            order_df['agent_type'] = order_df['agent_id'].map(agent_type_map)
            
            # Group orders by round, agent_type and decision
            grouped_orders = order_df.groupby(['round', 'agent_type', 'decision']).agg({
                'quantity': 'sum'
            }).reset_index()
            
            # Plot order flow with buys and sells clearly distinguished
            plt.figure(figsize=(12, 8))
            
            # Create a plot with two subplots - one for buys, one for sells
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Process buy orders
            buy_orders = grouped_orders[grouped_orders['decision'] == 'buy']
            buy_pivot = buy_orders.pivot(index='round', columns='agent_type', values='quantity').fillna(0)
            buy_pivot.plot(kind='bar', stacked=True, ax=ax1, alpha=0.7)
            ax1.set_title('Buy Order Volume by Agent Type')
            ax1.set_ylabel('Volume')
            ax1.legend(title='Agent Type')
            ax1.grid(True, alpha=0.3)
            
            # Process sell orders
            sell_orders = grouped_orders[grouped_orders['decision'] == 'sell']
            sell_pivot = sell_orders.pivot(index='round', columns='agent_type', values='quantity').fillna(0)
            sell_pivot.plot(kind='bar', stacked=True, ax=ax2, alpha=0.7)
            ax2.set_title('Sell Order Volume by Agent Type')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Volume')
            ax2.legend(title='Agent Type')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_plot_with_suffix('order_flow_by_type')
            
            # Add a net order flow plot
            plt.figure(figsize=(12, 6))
            
            # Convert sells to negative for net calculation
            grouped_orders_net = grouped_orders.copy()
            grouped_orders_net.loc[grouped_orders_net['decision'] == 'sell', 'quantity'] *= -1
            
            # Calculate net order flow
            net_orders = grouped_orders_net.groupby(['round', 'agent_type'])['quantity'].sum().unstack().fillna(0)
            
            # Plot net order flow
            net_orders.plot(kind='bar', figsize=(12, 6))
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Round')
            plt.ylabel('Net Order Flow (Positive = Net Buying, Negative = Net Selling)')
            plt.title('Net Order Flow by Agent Type')
            plt.legend(title='Agent Type')
            plt.grid(True, alpha=0.3)
            
            save_plot_with_suffix('order_flow_net')
            
            # Alternative implementation for the stacked area plot
            plt.figure(figsize=(12, 6))
            
            # Create a DataFrame with both buys (positive) and sells (negative)
            combined_orders = grouped_orders_net.pivot(index='round', columns='agent_type', values='quantity').fillna(0)
            
            # Plot as lines instead of areas for clarity
            combined_orders.plot(kind='line', marker='o', linewidth=2)
            
            plt.xlabel('Round')
            plt.ylabel('Order Volume (Positive = Buy, Negative = Sell)')
            plt.title('Net Order Flow by Agent Type')
            plt.legend(title='Agent Type')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            save_plot_with_suffix('order_flow_aggregated')
        except Exception as e:
            print(f"  Error creating order flow plots: {str(e)}")

def copy_data_to_latest(simulation):
    """Copy all relevant data files to latest_sim directory, organized by scenario"""
    latest_dir = Path('logs') / 'latest_sim'
    
    # Create latest_sim directory if it doesn't exist
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    # Debug print
    print(f"Copying data to latest_sim for scenario: {simulation.sim_type}")
    
    # Create a scenario-specific subdirectory
    scenario_dir = latest_dir / simulation.sim_type
    print(f"Scenario directory path: {scenario_dir}")
    
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # First check if we have source data to copy
    source_data_dir = simulation.run_dir / 'data'
    source_plots_dir = simulation.run_dir / 'plots'
    metadata_file = simulation.run_dir / 'metadata.json'
    params_file = simulation.run_dir / 'parameters.json'
    
    print(f"Source data directory: {source_data_dir} (exists: {source_data_dir.exists()})")
    print(f"Source plots directory: {source_plots_dir} (exists: {source_plots_dir.exists()})")
    
    # Handle data directory - copy files individually without removing directory
    if source_data_dir.exists() and any(source_data_dir.iterdir()):
        scenario_data_dir = scenario_dir / 'data'
        scenario_data_dir.mkdir(exist_ok=True)
        print(f"Created scenario data directory: {scenario_data_dir}")
        
        # Copy each file individually
        for file in source_data_dir.glob('*'):
            target_file = scenario_data_dir / file.name
            shutil.copy2(file, target_file)
            print(f"Copied {file} to {target_file}")
    
    # Handle plots directory - copy files individually without removing directory
    if source_plots_dir.exists() and any(source_plots_dir.iterdir()):
        scenario_plots_dir = scenario_dir / 'plots'
        scenario_plots_dir.mkdir(exist_ok=True)
        print(f"Created scenario plots directory: {scenario_plots_dir}")
        
        # Copy each file individually
        for file in source_plots_dir.glob('*'):
            target_file = scenario_plots_dir / file.name
            shutil.copy2(file, target_file)
            print(f"Copied {file} to {target_file}")
    
    # Copy metadata.json if it exists
    if metadata_file.exists():
        target_metadata = scenario_dir / 'metadata.json'
        shutil.copy2(metadata_file, target_metadata)
        print(f"Copied metadata to {target_metadata}")
    
    # Copy parameters.json if it exists
    if params_file.exists():
        target_params = scenario_dir / 'parameters.json'
        shutil.copy2(params_file, target_params)
        print(f"Copied parameters to {target_params}")

def run_scenario(
    scenario_name: str,
    allow_short_selling: bool = None,
    margin_requirement: float = None,
    borrow_rate: float = None,
):
    """Run a single scenario by name"""
    # Load scenario
    scenario = get_scenario(scenario_name)
    params = scenario.parameters

    # Apply overrides if provided
    agent_params = params.get("AGENT_PARAMS", {})
    if allow_short_selling is not None:
        agent_params['allow_short_selling'] = allow_short_selling
    if margin_requirement is not None:
        agent_params['margin_requirement'] = margin_requirement
    if borrow_rate is not None:
        borrow_model = agent_params.get('borrow_model', {})
        borrow_model['rate'] = borrow_rate
        borrow_model.setdefault('payment_frequency', 1)
        agent_params['borrow_model'] = borrow_model

    # Set random seeds for reproducibility
    np.random.seed(params["RANDOM_SEED"])
    random.seed(params["RANDOM_SEED"])

    # Create run directory with scenario info
    run_dir = create_run_directory(
        sim_type=scenario.name,
        description=scenario.description,
        parameters=params
    )

    # Check if this is a multi-stock scenario
    is_multi_stock = params.get("IS_MULTI_STOCK", False)

    if is_multi_stock:
        # Multi-stock scenario: pass stock_configs instead of single stock params
        simulation = BaseSimulation(
            num_rounds=params["NUM_ROUNDS"],
            initial_price=0,  # Unused for multi-stock, but required parameter
            fundamental_price=0,  # Unused for multi-stock, but required parameter
            redemption_value=None,
            transaction_cost=params.get("TRANSACTION_COST", 0.0),
            lendable_shares=params.get("LENDABLE_SHARES", 0),
            agent_params=params["AGENT_PARAMS"],
            dividend_params=None,  # Per-stock dividend params in stock_configs
            model_open_ai=params["MODEL_OPEN_AI"],
            interest_params=params["INTEREST_MODEL"],
            hide_fundamental_price=params["HIDE_FUNDAMENTAL_PRICE"],
            infinite_rounds=params["INFINITE_ROUNDS"],
            sim_type=scenario.name,
            stock_configs=params["STOCKS"]  # NEW: Pass stock configurations
        )
    else:
        # Single-stock scenario: original behavior (backwards compatible)
        redemption_value = params.get("REDEMPTION_VALUE", None)

        simulation = BaseSimulation(
            num_rounds=params["NUM_ROUNDS"],
            initial_price=params["INITIAL_PRICE"],
            fundamental_price=params["FUNDAMENTAL_PRICE"],
            redemption_value=redemption_value,
            transaction_cost=params["TRANSACTION_COST"],
            lendable_shares=params.get("LENDABLE_SHARES", 0),
            agent_params=params["AGENT_PARAMS"],
            dividend_params=params["DIVIDEND_PARAMS"],
            model_open_ai=params["MODEL_OPEN_AI"],
            interest_params=params["INTEREST_MODEL"],
            hide_fundamental_price=params["HIDE_FUNDAMENTAL_PRICE"],
            infinite_rounds=params["INFINITE_ROUNDS"],
            sim_type=scenario.name
        )

    # Save parameters and run simulation
    save_parameters(simulation.run_dir, params)
    simulation.run()
    save_plots(simulation, params)
    
    # Copy all data files to latest_sim
    copy_data_to_latest(simulation)

    # Print final agent states
    for agent_id in simulation.agent_repository.get_all_agent_ids():
        # Pass prices dict for multi-stock or single price for single-stock
        if simulation.is_multi_stock:
            prices = {stock_id: context.current_price for stock_id, context in simulation.contexts.items()}
        else:
            prices = simulation.context.current_price

        state = simulation.agent_repository.get_agent_state_snapshot(
            agent_id,
            prices
        )
        # Use wealth from snapshot (correctly calculated for both single and multi-stock)
        print(f"Agent {state.agent_id} Type: {state.agent_type} - "
            f"Cash: ${state.cash:.2f}, "
            f"Shares: {state.total_shares}, "
            f"Total Value: ${state.wealth:.2f}")

def main():
    """
    Main function to run simulations.
    Parses command-line arguments to run a specific scenario or list available ones.
    """
    import argparse
    
    # Don't convert warnings to errors - allow normal warnings
    np.seterr(all='warn')

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a trading simulation scenario.")
    parser.add_argument(
        "scenario", 
        nargs='?', 
        default=None, 
        help="The name of the scenario to run. If not provided, lists available scenarios."
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available scenarios and their descriptions."
    )
    parser.add_argument(
        "--allow-short-selling",
        dest="allow_short_selling",
        action="store_true",
        help="Enable short selling regardless of scenario settings"
    )
    parser.add_argument(
        "--disallow-short-selling",
        dest="allow_short_selling",
        action="store_false",
        help="Disable short selling regardless of scenario settings"
    )
    parser.set_defaults(allow_short_selling=None)
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=None,
        help="Override margin requirement for agents"
    )
    parser.add_argument(
        "--borrow-rate",
        type=float,
        default=None,
        help="Override borrow rate for short positions"
    )

    args = parser.parse_args()

    # Get available scenarios
    available_scenarios = list_scenarios()

    # If --list is used, print scenarios and exit
    if args.list:
        print("Available scenarios:")
        for name, desc in available_scenarios.items():
            print(f"  - {name}: {desc}")
        return

    # If no scenario is provided, print list and exit
    if args.scenario is None:
        print("No scenario specified. Please choose from the list below:")
        for name, desc in available_scenarios.items():
            print(f"  - {name}")
        print("\nUsage: python src/run_base_sim.py <scenario_name>")
        return

    # Check if the chosen scenario exists
    if args.scenario not in available_scenarios:
        print(f"Error: Scenario '{args.scenario}' not found.")
        print("Please choose from the available scenarios:")
        for name in available_scenarios.keys():
            print(f"  - {name}")
        return
        
    # Run the selected scenario
    scenario_name = args.scenario
    print(f"\nRunning scenario: {scenario_name}")
    print("-" * 50)
    try:
        run_scenario(
            scenario_name,
            allow_short_selling=args.allow_short_selling,
            margin_requirement=args.margin_requirement,
            borrow_rate=args.borrow_rate,
        )
        print(f"Successfully completed scenario: {scenario_name}")
    except Exception as e:
        print(f"Error running scenario {scenario_name}: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()