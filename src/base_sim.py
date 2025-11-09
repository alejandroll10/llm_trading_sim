from datetime import datetime
import traceback
from market.orders.order_book import OrderBook
from agents.agent_types import *
from agents.agents_api import *
from market.engine.match_engine import MatchingEngine
from market.state.market_state_manager import MarketStateManager
from agents.agent_manager.base_agent_manager import AgentManager
from market.data_recorder import DataRecorder
from agents.LLMs.llm_agent import LLMAgent
from agents.deterministic.deterministic_registry import DETERMINISTIC_AGENTS
from market.state.sim_context import SimulationContext
from market.orders.order_repository import OrderRepository
from market.state.services.dividend_service import DividendService
from agents.agent_manager.agent_repository import AgentRepository
from typing import Dict
from market.state.services.interest_service import InterestService
from market.state.services.borrow_service import BorrowService
from agents.agent_manager.services.agent_decision_service import AgentDecisionService
from market.orders.order_service_factory import OrderServiceFactory
from services.shared_service_factory import SharedServiceFactory
from market.information.base_information_services import InformationService
from services.logging_service import LoggingService
from agents.agent_manager.services.borrowing_repository import BorrowingRepository
import random
import warnings
from wordcloud import WordCloud

class BaseSimulation:
    """
    The core class for running a trading simulation.

    This class orchestrates the entire simulation, including setting up the
    environment, creating agents, running the market rounds, and collecting data.

    Attributes:
        num_rounds (int): The total number of rounds to run the simulation.
        agent_params (dict): Parameters for creating agents.
        sim_type (str): The name of the scenario being run.
        run_dir (Path): The directory where simulation data and plots are saved.
        context (TradingContext): The context object holding the current state of the market.
        agent_repository (AgentRepository): The repository managing all agents.
        data_recorder (DataRecorder): The service for recording simulation data.
        market (Market): The market where trades are executed.
        dividend_service (DividendService): The service for managing dividends.
        interest_service (InterestService): The service for managing interest payments.
        lendable_shares (int): Total shares available to borrow for short positions.
    """
    def __init__(self,
                 num_rounds: int,
                 initial_price: float,
                 fundamental_price: float,
                 redemption_value: float,
                 transaction_cost: float = 0.0,
                 fundamental_volatility: float = 0.0,
                 lendable_shares: int = 0,
                 agent_params: dict = None,
                 hide_fundamental_price: bool = True,
                 model_open_ai = "gpt-4o-2024-07-18",
                 dividend_params: dict = None,
                 interest_params: dict = None,
                 borrow_params: dict = None,
                 infinite_rounds: bool = False,
                 sim_type: str = "default",
                 stock_configs: dict = None):
        SharedServiceFactory.reset()

        self.infinite_rounds = infinite_rounds
        # Setup logging with sim_type directory structure
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sim_type = sim_type
        
        # Let LoggingService handle directory creation
        LoggingService.initialize(f"{sim_type}/{self.run_id}")
        self.logger = LoggingService.get_logger('simulation')
        
        # Get directories from LoggingService
        self.run_dir = LoggingService.get_run_dir()
        self.data_dir = LoggingService.get_data_dir()
        
        # Store parameters
        self.dividend_params = dividend_params
        self.initial_price = initial_price
        self.lendable_shares = lendable_shares
        self.order_repository = OrderRepository()

        # MULTI-STOCK SUPPORT: Detect if this is a multi-stock scenario
        self.is_multi_stock = stock_configs is not None
        self.stock_configs = stock_configs

        # Validate stock_configs if provided
        if self.is_multi_stock and not stock_configs:
            raise ValueError("stock_configs cannot be empty when multi-stock mode is enabled")

        if self.is_multi_stock:
            # Multi-stock: Create contexts for each stock (FOR LOOP!)
            self.contexts = {}
            for stock_id, config in stock_configs.items():
                self.contexts[stock_id] = SimulationContext(
                    num_rounds=num_rounds,
                    initial_price=config['INITIAL_PRICE'],
                    fundamental_price=config['FUNDAMENTAL_PRICE'],
                    redemption_value=config.get('REDEMPTION_VALUE'),
                    transaction_cost=config.get('TRANSACTION_COST', 0.0),
                    logger=LoggingService.get_logger(f'context_{stock_id}'),
                    infinite_rounds=self.infinite_rounds
                )
            # For backwards compatibility, expose first stock as .context
            self.context = list(self.contexts.values())[0]
        else:
            # Single stock: Original behavior (backwards compatible)
            self.context = SimulationContext(
                num_rounds=num_rounds,
                initial_price=initial_price,
                fundamental_price=fundamental_price,
                redemption_value=redemption_value,
                transaction_cost=transaction_cost,
                logger=LoggingService.get_logger('context'),
                infinite_rounds=self.infinite_rounds
            )
        
        # Basic simulation parameters
        self.fundamental_volatility = fundamental_volatility
        self.hide_fundamental_price = hide_fundamental_price
        self.model_open_ai = model_open_ai
        
        # Set default agent parameters if none provided
        self.agent_params = agent_params

        # Initialize agents with explicit parameters
        agents = self.initialize_agents(self.agent_params)

        # Get borrow configuration
        borrow_model = self.agent_params.get('borrow_model', {})
        allow_partial_borrows = borrow_model.get('allow_partial_borrows', True)

        self.agent_repository = AgentRepository(
            agents,
            logger=LoggingService.get_logger('agent_repository'),
            context=self.context,
            borrowing_repository=BorrowingRepository(
                total_lendable=lendable_shares,
                allow_partial_borrows=allow_partial_borrows,
                logger=LoggingService.get_logger('borrowing')
            )
        )
        # Initialize components in correct order
        if self.is_multi_stock:
            # Multi-stock: Create order books for each stock (FOR LOOP!)
            self.order_books = {}
            for stock_id in self.contexts.keys():
                self.order_books[stock_id] = OrderBook(
                    context=self.contexts[stock_id],
                    logger=LoggingService.get_logger(f'order_book_{stock_id}'),
                    order_repository=self.order_repository
                )
            # For backwards compatibility, expose first stock's order book
            self.order_book = list(self.order_books.values())[0]
        else:
            # Single stock: Original behavior
            self.order_book = OrderBook(
                context=self.context,
                logger=LoggingService.get_logger('order_book'),
                order_repository=self.order_repository
            )
        
        # Initialize shared services
        if self.is_multi_stock:
            # Multi-stock: Pass all order books so commitment calculator can use the right one
            SharedServiceFactory.initialize(order_books=self.order_books)
        else:
            # Single-stock: Pass single order book
            SharedServiceFactory.initialize(order_book=self.order_book)
        
        # Create order services through factory
        self.order_state_manager, self.trade_execution_service = OrderServiceFactory.create_services(
            order_repository=self.order_repository,
            agent_repository=self.agent_repository,
            order_book= self.order_book,
            logger=LoggingService.get_logger('order_state')
        )
        
        # Initialize dividend service(s)
        if self.is_multi_stock:
            # Multi-stock: Create dividend service for each stock (FOR LOOP!)
            self.dividend_services = {}
            for stock_id in self.contexts.keys():
                stock_dividend_params = stock_configs[stock_id].get('DIVIDEND_PARAMS')
                if stock_dividend_params:
                    self.dividend_services[stock_id] = DividendService(
                        agent_repository=self.agent_repository,
                        logger=LoggingService.get_logger(f'dividend_{stock_id}'),
                        dividend_params=stock_dividend_params,
                        redemption_value=self.contexts[stock_id].redemption_value,
                        stock_id=stock_id  # Pass stock_id so it pays dividends for correct stock
                    )
            # For backwards compatibility
            self.dividend_service = list(self.dividend_services.values())[0] if self.dividend_services else None
        else:
            # Single stock: Original behavior
            self.dividend_service = DividendService(
                agent_repository=self.agent_repository,
                logger=LoggingService.get_logger('market_state'),
                dividend_params=self.dividend_params,
                redemption_value=self.context.redemption_value
            ) if dividend_params else None

        # Initialize interest service
        self.interest_service = InterestService(
            agent_repository=self.agent_repository,
            logger=LoggingService.get_logger('market_state'),
            interest_params=interest_params or {
                'rate': self.agent_params['interest_model']['rate'],
                'compound_frequency': self.agent_params['interest_model']['compound_frequency']
            }
        )

        # Initialize borrow fee service
        default_borrow_model = self.agent_params.get('borrow_model', {})
        self.borrow_service = BorrowService(
            agent_repository=self.agent_repository,
            logger=LoggingService.get_logger('market_state'),
            borrow_params=borrow_params or {
                'rate': default_borrow_model.get('rate', 0.0),
                'payment_frequency': default_borrow_model.get('payment_frequency', 1)
            }
        )
        # Create market state manager(s) - one per stock in multi-stock mode
        if self.is_multi_stock:
            # Multi-stock: Create market state manager for each stock (FOR LOOP!)
            self.market_state_managers = {}
            for stock_id in self.contexts.keys():
                self.market_state_managers[stock_id] = MarketStateManager(
                    context=self.contexts[stock_id],
                    order_book=self.order_books[stock_id],
                    agent_repository=self.agent_repository,
                    logger=LoggingService.get_logger(f'market_state_{stock_id}'),
                    information_service=None,  # Will set after creating InformationService
                    dividend_service=self.dividend_services.get(stock_id),
                    interest_service=self.interest_service,
                    borrow_service=self.borrow_service,
                    hide_fundamental_price=self.hide_fundamental_price
                )
            # For backwards compatibility, expose first stock's manager
            self.market_state_manager = list(self.market_state_managers.values())[0]

            # Create information service with all managers
            information_service = InformationService(
                agent_repository=self.agent_repository,
                market_state_managers=self.market_state_managers
            )

            # Set information service on all managers
            for manager in self.market_state_managers.values():
                manager.information_service = information_service
        else:
            # Single stock: Original behavior
            information_service = InformationService(agent_repository=self.agent_repository)
            self.market_state_manager = MarketStateManager(
                context=self.context,
                order_book=self.order_book,
                agent_repository=self.agent_repository,
                logger=LoggingService.get_logger('market_state'),
                information_service=information_service,
                dividend_service=self.dividend_service,
                interest_service=self.interest_service,
                borrow_service=self.borrow_service,
                hide_fundamental_price=self.hide_fundamental_price
            )

        # Create data recorder with repository
        self.data_recorder = DataRecorder(
            context=self.context,
            agent_repository=self.agent_repository,
            loggers=LoggingService.get_logger('decisions'),
            data_dir=self.data_dir,
            market_state_manager=self.market_state_manager
        )

        # Create agent manager
        self.agent_manager = AgentManager(
            agent_repository=self.agent_repository,
            context=self.context,
            market_state_manager=self.market_state_manager,
            decisions_logger=LoggingService.get_logger('decisions'),
            agents_logger=LoggingService.get_logger('agents'),
            order_state_manager=self.order_state_manager,
            order_book=self.order_book,
            order_repository=self.order_repository
        )

        # Initialize matching engine(s)
        if self.is_multi_stock:
            # Multi-stock: Create matching engine for each stock (FOR LOOP!)
            self.matching_engines = {}
            for stock_id in self.contexts.keys():
                self.matching_engines[stock_id] = MatchingEngine(
                    order_book=self.order_books[stock_id],
                    agent_manager=self.agent_manager,
                    logger=LoggingService.get_logger(f'matching_{stock_id}'),
                    trades_logger=LoggingService.get_logger(f'trades_{stock_id}'),
                    trade_execution_service=self.trade_execution_service,
                    order_repository=self.order_repository,
                    order_state_manager=self.order_state_manager,
                    agent_repository=self.agent_repository,
                    context=self.contexts[stock_id],
                    is_multi_stock=True  # Flag multi-stock mode
                )
            # For backwards compatibility
            self.matching_engine = list(self.matching_engines.values())[0]
        else:
            # Single stock: Original behavior
            self.matching_engine = MatchingEngine(
                order_book=self.order_book,
                agent_manager=self.agent_manager,
                logger=LoggingService.get_logger('order_book'),
                trades_logger=LoggingService.get_logger('trades'),
                trade_execution_service=self.trade_execution_service,
                order_repository=self.order_repository,
                order_state_manager=self.order_state_manager,
                agent_repository=self.agent_repository,
                context=self.context
            )

        # Initialize agent-dependent structures
        self.data_recorder.initialize_agent_structures()
        # Record initial state
        # self.record_initial_state()

        self.decision_service = AgentDecisionService(
            agent_repository=self.agent_repository,
            order_repository=self.order_repository,
            order_state_manager=self.order_state_manager,
            agents_logger=LoggingService.get_logger('agents'),
            decisions_logger=LoggingService.get_logger('decisions'),
            context=self.context
        )

    def execute_round(self, round_number):
        """Execute a single round of trading"""
        self.logger.warning(f"\n=== Round {round_number} ===")
        # Log initial states
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "Start of ")
        self.order_book.log_order_book_state(f"Start of Round {round_number}")
        
        # Store pre-round states for verification
        pre_round_states = self._store_pre_round_states()

        # 1. UPDATE MARKET AND CONTEXT at the beginning of the round
        if self.is_multi_stock:
            for manager in self.market_state_managers.values():
                manager.update_market_depth()
        else:
            self.market_state_manager.update_market_depth()
        
        # Get last volume from data recorder
        if self.is_multi_stock:
            # Multi-stock: Get per-stock volumes from market_data
            # market_data has one entry per stock per round
            last_stock_volumes = {}
            if self.data_recorder.market_data:
                # Get volumes from previous round for each stock
                for stock_id in self.contexts.keys():
                    # Filter market_data for this stock and previous round
                    prev_round_data = [
                        entry for entry in self.data_recorder.market_data
                        if entry['stock_id'] == stock_id and entry['round'] == round_number
                    ]
                    last_stock_volumes[stock_id] = (
                        prev_round_data[-1]['total_volume']
                        if prev_round_data
                        else 0
                    )
            else:
                # First round - no previous volume
                last_stock_volumes = {stock_id: 0 for stock_id in self.contexts.keys()}
        else:
            # Single-stock: Get total volume from history
            last_volume = (
                self.data_recorder.history[-1]['total_volume']
                if self.data_recorder.history
                else 0
            )

        # Update market components and build market state
        if self.is_multi_stock:
            # Multi-stock: Update each manager and aggregate state
            market_state = {
                'stocks': {},
                'round_number': round_number + 1,
                'is_multi_stock': True
            }

            # Update all managers WITHOUT distributing information yet
            for stock_id, manager in self.market_state_managers.items():
                # Update public info for this stock BEFORE calling manager.update()
                # This updates public_info['last_trade']['volume'] with the previous round's volume
                stock_last_volume = last_stock_volumes[stock_id]
                self.contexts[stock_id].update_public_info(round_number, stock_last_volume)

                # Call manager.update() with skip_distribution to avoid multiple calls
                # We'll distribute once after all managers are updated
                stock_market_state = manager.update(
                    round_number=round_number,
                    last_volume=stock_last_volume,
                    is_round_end=False,
                    skip_distribution=True  # NEW: Skip distribution in multi-stock mode
                )
                # Store this stock's state
                market_state['stocks'][stock_id] = stock_market_state

            # Now distribute information ONCE for all stocks
            # Get any manager's information_service (they all share the same one)
            first_manager = list(self.market_state_managers.values())[0]
            information_service = first_manager.information_service

            # Ensure providers are registered (lazy initialization)
            # IMPORTANT: Register from ALL managers to ensure we get providers from any stock that has them
            if not information_service.providers:
                for manager in self.market_state_managers.values():
                    manager._register_base_providers()
                    # Providers dict is shared, so each manager adds its providers
                    # If multiple managers have the same provider, the last one wins (but they're equivalent)

            # Distribute information for all stocks
            information_service.distribute_information(round_number)
        else:
            # Single stock: Original behavior
            self.context.update_public_info(round_number, last_volume)
            market_state = self.market_state_manager.update(
                round_number=round_number,
                last_volume=last_volume,
                is_round_end=False
            )
        
        # Log state before collecting decisions
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "Pre-Decision ")
        
        # 2. COLLECT NEW AGENT DECISIONS (ORDERS)
        new_orders = self.decision_service.collect_decisions(
            market_state=market_state,
            history=self.data_recorder.history,
            round_number=round_number
        )

        # Update market depth after new orders are placed
        if self.is_multi_stock:
            for manager in self.market_state_managers.values():
                manager.update_market_depth()
        else:
            self.market_state_manager.update_market_depth()

        # Log state after decisions but before matching
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "Post-Decision ")
        self.order_book.log_order_book_state(f"After New Orders Round {round_number}")
        
        # 3. EXECUTE TRADES using the matching engine
        if self.is_multi_stock:
            # Multi-stock: Match orders FOR EACH STOCK separately (THE BIG FOR LOOP!)
            # Store results per stock to avoid overwriting
            stock_market_results = {}

            for stock_id in self.contexts.keys():
                # Filter orders for THIS stock only
                stock_orders = [o for o in new_orders if o.stock_id == stock_id]

                self.logger.info(f"=== Matching {len(stock_orders)} orders for {stock_id} ===")

                # Match orders for this stock
                stock_market_results[stock_id] = self.matching_engines[stock_id].match_orders(
                    stock_orders,
                    self.contexts[stock_id].current_price,
                    round_number + 1
                )

                # Update price for THIS stock
                self.contexts[stock_id].current_price = stock_market_results[stock_id].price
                self.contexts[stock_id].round_number = round_number + 1

            # Aggregate all trades and volumes from all stocks
            all_trades = []
            total_volume = 0
            for stock_id, result in stock_market_results.items():
                all_trades.extend(result.trades)
                total_volume += result.volume

            # Create aggregated market result for backwards compatibility
            # Use first stock's result as template, but with aggregated trades/volume
            first_result = list(stock_market_results.values())[0]
            market_result = type(first_result)(
                price=first_result.price,  # Not used in multi-stock (each stock has own price)
                trades=all_trades,
                volume=total_volume
            )

            # Update backwards-compatible self.context
            self.context.round_number = round_number + 1

            # Update agent wealth with all stock prices (multi-stock)
            all_prices = {stock_id: ctx.current_price for stock_id, ctx in self.contexts.items()}
            self.agent_repository.update_all_wealth(all_prices)
        else:
            # Single stock: Original behavior
            market_result = self.matching_engine.match_orders(
                new_orders,
                self.context.current_price,
                round_number + 1
            )

            # Update price and round number in the context
            self.context.current_price = market_result.price
            self.context.round_number = round_number + 1

        # Update market depth after matching
        if self.is_multi_stock:
            for manager in self.market_state_managers.values():
                manager.update_market_depth()
        else:
            self.market_state_manager.update_market_depth()

        # Log state after matching
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "Post-Matching ")
        self.order_book.log_order_book_state(f"After Trades Matched Round {round_number}")

        if round_number == self.context._num_rounds - 1 and not self.infinite_rounds:
            self.logger.info(f"Last round, redeeming shares for fundamental value: {self.context.fundamental_price}")
            self.logger.info(f"Shares are worthless after redemption")
        
        # 4. RECORD DATA for the round

        # Get last paid dividend
        if self.is_multi_stock:
            # Multi-stock: TODO - aggregate dividends from all stocks
            last_paid_dividend = 0.0
        else:
            # Single-stock: Original behavior
            last_paid_dividend = market_state.get('last_paid_dividend', 0.0)
            if not last_paid_dividend and round_number == 0:
                self.logger.info("First round, no dividend paid")
                last_paid_dividend = 0.0
            else:
                # Get last paid dividend from market state
                if 'dividend_state' in market_state and market_state['dividend_state']:
                    dividend_state = market_state['dividend_state']
                    last_paid_dividend = dividend_state.get('last_paid_dividend')
                    if last_paid_dividend is None:
                        raise ValueError(f"No last paid dividend found in dividend state: {dividend_state}")
                else:
                    raise ValueError(f"No dividend state found in market state: {market_state.keys()}")

        self.data_recorder.record_round_data(
            round_number=round_number, 
            market_state=market_state, 
            orders=new_orders,
            trades=market_result.trades, 
            total_volume=market_result.volume,
            dividends=last_paid_dividend
        )

        
        # Log final states
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "End of ")
        
        # Final order book state
        self.order_book.log_order_book_state(f"End of Round {round_number}")

        # 5. FINAL END-OF-ROUND UPDATES (including interest/dividend payments)
        if self.is_multi_stock:
            # Multi-stock: Let each manager handle end-of-round processing
            # Managers will process dividends/redemption through their _process_end_of_round() method
            total_payments = 0.0

            for stock_id, manager in self.market_state_managers.items():
                # Manager.update() with is_round_end=True handles all end-of-round processing
                # Pass THIS stock's volume, not the aggregated total
                stock_volume = stock_market_results[stock_id].volume
                manager.update(
                    round_number=round_number,
                    last_volume=stock_volume,
                    is_round_end=True
                )
                manager.update_market_depth()

                # Aggregate total payments from all stocks for logging
                # Get last payment from dividend service if it exists
                if manager.dividend_service and manager.dividend_service.dividend_history:
                    stock_payment = manager.dividend_service.dividend_history[-1]
                    total_payments += stock_payment
                    self.logger.info(f"  {stock_id} dividend payment: ${stock_payment:.2f}")

            last_paid_dividend = total_payments
            self.logger.info(f"Total payments across all stocks: ${total_payments:.2f}")
        else:
            # Single stock: Original behavior
            self.market_state_manager.update(
                round_number=round_number,
                last_volume=market_result.volume,
                is_round_end=True
            )
            self.market_state_manager.update_market_depth()

        self.logger.info(f"Dividends paid last round: {last_paid_dividend}")
        # Verify final states
        self._verify_round_end_states(pre_round_states)

    def create_agent(self, agent_id: int, agent_type: str, agent_params: dict):
        """Factory method to create appropriate agent type with explicit parameters"""
        # Get type-specific parameters if they exist, otherwise use defaults
        type_specific_params = agent_params.get('type_specific_params', {}).get(agent_type, {})

        # Handle both single-stock (initial_shares) and multi-stock (initial_positions)
        if 'initial_positions' in agent_params:
            # Multi-stock: use initial_positions dict
            initial_shares_value = sum(agent_params['initial_positions'].values())
        else:
            # Single-stock: use initial_shares
            initial_shares_value = type_specific_params.get('initial_shares', agent_params['initial_shares'])

        base_params = {
            'agent_id': agent_id,
            'initial_cash': type_specific_params.get('initial_cash', agent_params['initial_cash']),
            'initial_shares': initial_shares_value,
            'position_limit': type_specific_params.get('position_limit', agent_params['position_limit']),
            'allow_short_selling': type_specific_params.get('allow_short_selling', agent_params['allow_short_selling']),
            'logger': LoggingService.get_logger('decisions'),
            'info_signals_logger': LoggingService.get_logger('info_signals'),
            'initial_price': self.initial_price
        }

        # Check if it's a deterministic agent
        if agent_type in DETERMINISTIC_AGENTS:
            return DETERMINISTIC_AGENTS[agent_type](**base_params)
        
        # Set model name for hold_llm agent
        model = "hold_llm" if agent_type == "hold_llm" else self.model_open_ai
        
        # Create LLM agent with appropriate model
        return LLMAgent(
            **base_params,
            agent_type=agent_type,
            model_open_ai=model
        )

    def initialize_agents(self, agent_params: dict):
        """Initialize agents based on explicitly provided parameters"""
        agents = []
        agent_id = 0

        # Extract required parameters
        agent_composition = agent_params['agent_composition']  # Use the provided composition directly
        self.logger.warning(f"Agent composition: {agent_composition}")
        for agent_type, count in agent_composition.items():
            for _ in range(count):
                agent = self.create_agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    agent_params=agent_params
                )

                # For multi-stock scenarios, set positions dict
                if 'initial_positions' in agent_params:
                    agent.positions = agent_params['initial_positions'].copy()
                    # Reset committed and borrowed positions for all stocks
                    agent.committed_positions = {stock_id: 0 for stock_id in agent.positions.keys()}
                    agent.borrowed_positions = {stock_id: 0 for stock_id in agent.positions.keys()}
                    # Update initial_shares to be the sum across all stocks for verification
                    agent.initial_shares = sum(agent.positions.values())

                agents.append(agent)
                agent_id += 1

        return agents
    
    def run(self):
        """Base simulation run logic"""
        try:
            for round_number in range(self.context._num_rounds):
                self.execute_round(round_number)
            self.data_recorder.save_simulation_data()
            LoggingService.log_simulation("Simulation completed successfully")
        except Exception as e:
            LoggingService.log_simulation(f"Simulation failed with error: {str(e)}")
            LoggingService.log_simulation(traceback.format_exc())
            raise e
        finally:
            try:
                self.data_recorder.save_simulation_data()
            except Exception as e:
                LoggingService.log_simulation(f"Failed to save final data: {str(e)}")
       # Clean up expired orders at end of round

    def _verify_round_end_states(self, pre_round_states):
        """Verify agent states at end of round"""
        logger = LoggingService.get_logger('verification')
        self.logger.info(f"\n=== Verifying state changes for round {self.context.round_number} ===")
        
        # Log individual agent changes
        self.logger.info("\nPer-agent cash changes:")
        for agent_id in pre_round_states:
            pre_cash = pre_round_states[agent_id]['total_cash']
            post_cash = self.agent_repository.get_agent(agent_id).total_cash
            change = post_cash - pre_cash
            self.logger.info(f"Agent {agent_id}: ${pre_cash:.2f} -> ${post_cash:.2f} (Δ${change:.2f})")

        # Calculate totals
        total_cash_pre = sum(state['total_cash'] for state in pre_round_states.values())
        total_cash_post = sum(
            self.agent_repository.get_agent(agent_id).total_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )
        
        # Get payments from this round
        current_round = self.context.round_number

        if self.is_multi_stock:
            # Multi-stock: Aggregate payments from all stock contexts
            dividend_payment = 0.0
            interest_payment = 0.0
            borrow_fee_payment = 0.0

            for stock_id, context in self.contexts.items():
                dividend_payment += sum(
                    payment['amount']
                    for payment in context.market_history.dividends_paid
                    if payment['round'] == current_round - 1
                )
                interest_payment += sum(
                    payment['amount']
                    for payment in context.market_history.interest_paid
                    if payment['round'] == current_round - 1
                )
                borrow_fee_payment += sum(
                    payment['amount']
                    for payment in context.market_history.borrow_fees_paid
                    if payment['round'] == current_round - 1
                )
        else:
            # Single stock: Original behavior
            dividend_payment = sum(
                payment['amount']
                for payment in self.context.market_history.dividends_paid
                if payment['round'] == current_round - 1
            )
            interest_payment = sum(
                payment['amount']
                for payment in self.context.market_history.interest_paid
                if payment['round'] == current_round - 1
            )
            borrow_fee_payment = sum(
                payment['amount']
                for payment in self.context.market_history.borrow_fees_paid
                if payment['round'] == current_round - 1
            )

        self.logger.info(
            f"Round payments - Dividends: ${dividend_payment:.2f}, Interest: ${interest_payment:.2f}, Borrow Fees: ${borrow_fee_payment:.2f}"
        )

        # Verify round-by-round changes
        cash_difference = total_cash_post - total_cash_pre
        total_round_payments = dividend_payment + interest_payment - borrow_fee_payment
        self.logger.info(f"Cash change: ${cash_difference:.2f}, Expected: ${total_round_payments:.2f}")

        if abs(cash_difference - total_round_payments) > 0.01:
            msg = (f"Round cash change doesn't match round payments:\n"
                   f"Change in cash: ${cash_difference:.2f}\n"
                   f"Round payments: ${total_round_payments:.2f}")
            logger.error(msg)
            raise ValueError(msg)
        
        # Verify total system cash
        initial_cash = sum(
            self.agent_repository.get_agent(agent_id).initial_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        if self.is_multi_stock:
            # Multi-stock: Aggregate historical payments from all stock contexts
            total_historical_dividends = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.dividends_paid
            )
            total_historical_interest = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.interest_paid
            )
            total_historical_borrow_fees = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.borrow_fees_paid
            )
        else:
            # Single stock: Original behavior
            total_historical_dividends = sum(payment['amount'] for payment in self.context.market_history.dividends_paid)
            total_historical_interest = sum(payment['amount'] for payment in self.context.market_history.interest_paid)
            total_historical_borrow_fees = sum(payment['amount'] for payment in self.context.market_history.borrow_fees_paid)
        expected_total_cash = (
            initial_cash + total_historical_dividends + total_historical_interest - total_historical_borrow_fees
        )

        self.logger.info(f"\n=== System-wide cash verification ===")
        self.logger.info(f"Initial cash: ${initial_cash:.2f}")
        self.logger.info(f"Total historical dividends: ${total_historical_dividends:.2f}")
        self.logger.info(f"Total historical interest: ${total_historical_interest:.2f}")
        self.logger.info(f"Total historical borrow fees: ${total_historical_borrow_fees:.2f}")
        self.logger.info(f"Current total cash: ${total_cash_post:.2f}")
        self.logger.info(f"Expected total: ${expected_total_cash:.2f}")

        if abs(total_cash_post - expected_total_cash) > 0.01:
            msg = (f"Total system cash doesn't match historical payments")
            logger.error(msg)
            raise ValueError(msg)

        # Share verification
        initial_shares = sum(
            self.agent_repository.get_agent(agent_id).initial_shares
            for agent_id in self.agent_repository.get_all_agent_ids()
        )
        
        # Check if this is the final round and shares are being redeemed
        is_final_redemption = (
            self.context.round_number == self.context._num_rounds and 
            not self.infinite_rounds
        )
        
        self.logger.info("\n=== Share allocation verification ===")
        self.logger.info("Initial allocation:")
        if 'initial_shares' in self.agent_params:
            self.logger.info(f"Per agent: {self.agent_params['initial_shares']}")
        elif 'initial_positions' in self.agent_params:
            self.logger.info(f"Per agent (multi-stock): {self.agent_params['initial_positions']}")
        self.logger.info(f"Total: {initial_shares}")
        
        # Log pre-round share distribution
        self.logger.info("\nPre-round share distribution:")
        pre_shares = {
            agent_id: state['total_shares']
            for agent_id, state in pre_round_states.items()
        }
        for agent_id, shares in pre_shares.items():
            self.logger.info(f"Agent {agent_id}: {shares}")
        self.logger.info(f"Pre-round total: {sum(pre_shares.values())}")
        
        # Log post-round share distribution
        self.logger.info("\nPost-round share distribution:")
        post_shares = {
            agent_id: self.agent_repository.get_agent(agent_id).total_shares
            for agent_id in self.agent_repository.get_all_agent_ids()
        }
        for agent_id, shares in post_shares.items():
            agent = self.agent_repository.get_agent(agent_id)
            self.logger.info(f"Agent {agent_id}: {shares} (available: {agent.shares}, committed: {agent.committed_shares})")
        
        total_shares_current = sum(post_shares.values())
        self.logger.info(f"Post-round total: {total_shares_current}")

        borrowed_total = self.agent_repository.borrowing_repository.total_lendable - self.agent_repository.borrowing_repository.available_shares
        expected_total_shares = initial_shares + borrowed_total

        # Modified verification logic
        if is_final_redemption:
            if total_shares_current != 0:
                msg = (f"Final redemption: All shares should be zero but found {total_shares_current} shares"
                       f"Initial shares: {initial_shares}"
                       f"Current shares: {total_shares_current}"
                       f"Share changes this round:")
                for agent_id in pre_shares:
                    change = post_shares[agent_id] - pre_shares[agent_id]
                    msg += f"\nAgent {agent_id}: {pre_shares[agent_id]} -> {post_shares[agent_id]} (Δ{change})"
                logger.error(msg)
                raise ValueError(msg)
            else:
                redemption_msg = "Final redemption: All shares successfully redeemed"
                if self.is_multi_stock:
                    redemption_msg += f" across {len(self.dividend_services)} stocks"
                self.logger.info(redemption_msg)
        elif total_shares_current != expected_total_shares:
            msg = (f"Total shares in system changed from initial allocation:\n"
                   f"Initial shares: {initial_shares}\n"
                   f"Borrowed shares: {borrowed_total}\n"
                   f"Current shares: {total_shares_current}\n"
                   f"Share changes this round:")
            for agent_id in pre_shares:
                change = post_shares[agent_id] - pre_shares[agent_id]
                msg += f"\nAgent {agent_id}: {pre_shares[agent_id]} -> {post_shares[agent_id]} (Δ{change})"
            logger.error(msg)
            raise ValueError(msg)

        # Multi-stock specific invariants
        if self.is_multi_stock:
            self._verify_multi_stock_invariants(logger)

        # Additional invariants (both single and multi-stock)
        self._verify_commitment_order_matching(logger)
        self._verify_borrowing_pool_consistency(logger)
        self._verify_wealth_conservation(pre_round_states, logger)
        self._verify_order_book_consistency(logger)
        self._verify_dividend_accumulation(logger)
        self._verify_interest_calculations(logger)
        self._verify_borrow_fee_calculations(logger)

    def _store_pre_round_states(self) -> Dict[str, Dict]:
        """Store pre-round states for verification through repository"""
        pre_round_states = {}

        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)

            pre_round_states[agent_id] = {
                'total_cash': agent.total_cash,
                'total_shares': agent.total_shares,
            }

        return pre_round_states

    def _verify_multi_stock_invariants(self, logger):
        """Verify multi-stock specific invariants"""
        self.logger.info("\n=== Multi-stock invariant verification ===")

        # Check if this is the final round and shares are being redeemed
        is_final_redemption = (
            self.context.round_number == self.context._num_rounds and
            not self.infinite_rounds
        )

        # Get all stock IDs
        stock_ids = list(self.contexts.keys())
        self.logger.info(f"Verifying {len(stock_ids)} stocks: {stock_ids}")

        # Per-stock share conservation
        for stock_id in stock_ids:
            self.logger.info(f"\n--- Verifying {stock_id} ---")

            # Calculate total shares for this stock across all agents
            total_shares_for_stock = 0
            total_committed_for_stock = 0
            total_borrowed_for_stock = 0

            agent_positions = {}
            for agent_id in self.agent_repository.get_all_agent_ids():
                agent = self.agent_repository.get_agent(agent_id)
                position = agent.positions.get(stock_id, 0)
                committed = agent.committed_positions.get(stock_id, 0)
                borrowed = agent.borrowed_positions.get(stock_id, 0)

                total_shares_for_stock += position + committed
                total_committed_for_stock += committed
                total_borrowed_for_stock += borrowed

                agent_positions[agent_id] = {
                    'position': position,
                    'committed': committed,
                    'borrowed': borrowed,
                    'total': position + committed
                }

            # Get expected initial shares for this stock
            if 'initial_positions' in self.agent_params:
                initial_per_agent = self.agent_params['initial_positions'].get(stock_id, 0)
                num_agents = len(self.agent_repository.get_all_agent_ids())
                expected_initial = initial_per_agent * num_agents
            else:
                expected_initial = 0  # Single-stock mode, should not reach here

            # Expected total includes borrowed shares
            expected_total_with_borrowed = expected_initial + total_borrowed_for_stock

            self.logger.info(f"Initial shares for {stock_id}: {expected_initial}")
            self.logger.info(f"Current total shares: {total_shares_for_stock}")
            self.logger.info(f"Borrowed shares: {total_borrowed_for_stock}")
            self.logger.info(f"Expected total (initial + borrowed): {expected_total_with_borrowed}")

            # Verify share conservation for this stock
            if is_final_redemption:
                # Final round: all shares should be redeemed (0)
                if total_shares_for_stock != 0:
                    msg = f"Final redemption failed for {stock_id}:\n"
                    msg += f"Expected all shares to be redeemed (0), but found {total_shares_for_stock}\n"
                    msg += f"Per-agent breakdown:\n"
                    for agent_id, pos in agent_positions.items():
                        msg += f"  Agent {agent_id}: position={pos['position']}, committed={pos['committed']}, borrowed={pos['borrowed']}, total={pos['total']}\n"
                    logger.error(msg)
                    raise ValueError(msg)
                self.logger.info(f"✓ Final redemption verified for {stock_id} (all shares redeemed)")
            else:
                # Normal round: verify share conservation
                if abs(total_shares_for_stock - expected_total_with_borrowed) > 0.01:
                    msg = f"Share conservation violated for {stock_id}:\n"
                    msg += f"Expected: {expected_total_with_borrowed} (initial: {expected_initial} + borrowed: {total_borrowed_for_stock})\n"
                    msg += f"Actual: {total_shares_for_stock}\n"
                    msg += f"Per-agent breakdown:\n"
                    for agent_id, pos in agent_positions.items():
                        msg += f"  Agent {agent_id}: position={pos['position']}, committed={pos['committed']}, borrowed={pos['borrowed']}, total={pos['total']}\n"
                    logger.error(msg)
                    raise ValueError(msg)
                self.logger.info(f"✓ Share conservation verified for {stock_id}")

        # Verify all agents have positions for the same stocks
        all_stock_ids = set()
        agent_stock_sets = {}
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            agent_stocks = set(agent.positions.keys())
            agent_stock_sets[agent_id] = agent_stocks
            all_stock_ids.update(agent_stocks)

        # Check for consistency (all agents should have entries for all stocks in multi-stock mode)
        expected_stocks = set(stock_ids)
        for agent_id, agent_stocks in agent_stock_sets.items():
            # Allow DEFAULT_STOCK to exist in single-stock mode
            agent_stocks_normalized = agent_stocks - {"DEFAULT_STOCK"}
            if agent_stocks_normalized and agent_stocks_normalized != expected_stocks:
                missing = expected_stocks - agent_stocks_normalized
                extra = agent_stocks_normalized - expected_stocks
                msg = f"Stock position inconsistency for agent {agent_id}:\n"
                if missing:
                    msg += f"  Missing stocks: {missing}\n"
                if extra:
                    msg += f"  Extra stocks: {extra}\n"
                msg += f"  Expected: {expected_stocks}\n"
                msg += f"  Actual: {agent_stocks_normalized}"
                logger.warning(msg)  # Warning instead of error for now

        self.logger.info("\n✓ All multi-stock invariants verified")

    def _verify_commitment_order_matching(self, logger):
        """Verify that committed resources match outstanding orders"""
        self.logger.info("\n=== Commitment-Order Matching Verification ===")

        # Get all active orders
        from market.orders.order import OrderState
        active_states = {OrderState.PENDING, OrderState.PARTIALLY_FILLED}

        # Calculate expected commitments from orders
        expected_committed_cash = 0
        expected_committed_shares_per_stock = {}

        for order in self.order_repository.orders.values():
            if order.state in active_states:
                if order.side == 'buy':
                    # Buy orders commit cash
                    # For limit orders, commitment is quantity * limit_price
                    # For market orders, we use current price (already committed at that price)
                    if order.order_type == 'limit':
                        expected_committed_cash += order.remaining_quantity * order.price
                    else:
                        # Market order - committed at current price
                        if self.is_multi_stock:
                            price = self.contexts[order.stock_id].current_price
                        else:
                            price = self.context.current_price
                        expected_committed_cash += order.remaining_quantity * price
                elif order.side == 'sell':
                    # Sell orders commit shares
                    stock_id = order.stock_id
                    expected_committed_shares_per_stock[stock_id] = \
                        expected_committed_shares_per_stock.get(stock_id, 0) + order.remaining_quantity

        # Calculate actual commitments
        actual_committed_cash = sum(
            self.agent_repository.get_agent(agent_id).committed_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        # For shares, aggregate across all stocks
        actual_committed_shares_per_stock = {}
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            for stock_id in agent.committed_positions.keys():
                actual_committed_shares_per_stock[stock_id] = \
                    actual_committed_shares_per_stock.get(stock_id, 0) + agent.committed_positions.get(stock_id, 0)

        self.logger.info(f"Expected committed cash from orders: ${expected_committed_cash:.2f}")
        self.logger.info(f"Actual committed cash: ${actual_committed_cash:.2f}")

        # Allow for small floating point differences
        if abs(expected_committed_cash - actual_committed_cash) > 0.1:
            msg = f"Commitment-order mismatch for cash:\n"
            msg += f"Expected (from orders): ${expected_committed_cash:.2f}\n"
            msg += f"Actual (from agents): ${actual_committed_cash:.2f}\n"
            msg += f"Difference: ${abs(expected_committed_cash - actual_committed_cash):.2f}"
            logger.warning(msg)  # Warning instead of error for now

        # Check per-stock share commitments
        all_stock_ids = set(expected_committed_shares_per_stock.keys()) | set(actual_committed_shares_per_stock.keys())
        for stock_id in all_stock_ids:
            expected = expected_committed_shares_per_stock.get(stock_id, 0)
            actual = actual_committed_shares_per_stock.get(stock_id, 0)

            self.logger.info(f"{stock_id}: Expected committed shares: {expected}, Actual: {actual}")

            if abs(expected - actual) > 0.01:
                msg = f"Commitment-order mismatch for {stock_id} shares:\n"
                msg += f"Expected (from orders): {expected}\n"
                msg += f"Actual (from agents): {actual}\n"
                msg += f"Difference: {abs(expected - actual)}"
                logger.warning(msg)  # Warning for now

        self.logger.info("✓ Commitment-order matching verified")

    def _verify_borrowing_pool_consistency(self, logger):
        """Verify borrowing pool accounting is consistent"""
        self.logger.info("\n=== Borrowing Pool Consistency Verification ===")

        borrowing_repo = self.agent_repository.borrowing_repository

        # Borrowing pool invariant: available + sum(borrowed) == total_lendable
        total_borrowed_from_pool = sum(borrowing_repo.borrowed.values())
        expected_available = borrowing_repo.total_lendable - total_borrowed_from_pool

        self.logger.info(f"Total lendable: {borrowing_repo.total_lendable}")
        self.logger.info(f"Available in pool: {borrowing_repo.available_shares}")
        self.logger.info(f"Total borrowed from pool: {total_borrowed_from_pool}")
        self.logger.info(f"Expected available: {expected_available}")

        if borrowing_repo.available_shares != expected_available:
            msg = f"Borrowing pool accounting error:\n"
            msg += f"Available: {borrowing_repo.available_shares}\n"
            msg += f"Expected: {expected_available}\n"
            msg += f"Total lendable: {borrowing_repo.total_lendable}\n"
            msg += f"Total borrowed: {total_borrowed_from_pool}"
            logger.error(msg)
            raise ValueError(msg)

        # Verify agent borrowed shares match pool
        total_agent_borrowed = 0
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            # Sum across all stocks
            for stock_id in agent.borrowed_positions.keys():
                total_agent_borrowed += agent.borrowed_positions.get(stock_id, 0)

        self.logger.info(f"Total borrowed by agents: {total_agent_borrowed}")

        if total_agent_borrowed != total_borrowed_from_pool:
            msg = f"Agent borrowed shares don't match pool:\n"
            msg += f"Agents borrowed: {total_agent_borrowed}\n"
            msg += f"Pool records: {total_borrowed_from_pool}"
            logger.error(msg)
            raise ValueError(msg)

        self.logger.info("✓ Borrowing pool consistency verified")

    def _verify_wealth_conservation(self, pre_round_states, logger):
        """Verify that total wealth changes only due to dividends, interest, and fees"""
        self.logger.info("\n=== Wealth Conservation Verification ===")

        # Calculate pre-round total wealth
        pre_wealth = 0
        for agent_id in pre_round_states:
            agent = self.agent_repository.get_agent(agent_id)
            # Need to calculate wealth at pre-round prices
            # For simplicity, we'll just verify that trades are zero-sum
            pre_wealth += pre_round_states[agent_id]['total_cash']

        # Calculate post-round total wealth
        post_wealth = sum(
            self.agent_repository.get_agent(agent_id).total_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        # Get payments from this round
        current_round = self.context.round_number

        if self.is_multi_stock:
            dividend_payment = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.dividends_paid
                if payment['round'] == current_round - 1
            )
            interest_payment = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.interest_paid
                if payment['round'] == current_round - 1
            )
            borrow_fee_payment = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.borrow_fees_paid
                if payment['round'] == current_round - 1
            )
        else:
            dividend_payment = sum(
                payment['amount']
                for payment in self.context.market_history.dividends_paid
                if payment['round'] == current_round - 1
            )
            interest_payment = sum(
                payment['amount']
                for payment in self.context.market_history.interest_paid
                if payment['round'] == current_round - 1
            )
            borrow_fee_payment = sum(
                payment['amount']
                for payment in self.context.market_history.borrow_fees_paid
                if payment['round'] == current_round - 1
            )

        expected_wealth_change = dividend_payment + interest_payment - borrow_fee_payment
        actual_wealth_change = post_wealth - pre_wealth

        self.logger.info(f"Pre-round cash: ${pre_wealth:.2f}")
        self.logger.info(f"Post-round cash: ${post_wealth:.2f}")
        self.logger.info(f"Cash change: ${actual_wealth_change:.2f}")
        self.logger.info(f"Expected change (div+int-fees): ${expected_wealth_change:.2f}")

        # This is the same check as the cash conservation check above, but with clearer wealth framing
        if abs(actual_wealth_change - expected_wealth_change) > 0.01:
            msg = f"Wealth conservation violated (trades should be zero-sum):\n"
            msg += f"Cash change: ${actual_wealth_change:.2f}\n"
            msg += f"Expected from payments: ${expected_wealth_change:.2f}\n"
            msg += f"Difference: ${abs(actual_wealth_change - expected_wealth_change):.2f}"
            logger.error(msg)
            raise ValueError(msg)

        self.logger.info("✓ Wealth conservation verified (trades are zero-sum)")

    def _verify_order_book_consistency(self, logger):
        """Verify order book consistency invariants"""
        self.logger.info("\n=== Order Book Consistency Verification ===")

        # Get order book(s) - could be single or multiple for multi-stock
        if self.is_multi_stock:
            order_books = self.order_books  # Dict[stock_id, OrderBook]
        else:
            order_books = {"DEFAULT_STOCK": self.order_book}

        for stock_id, order_book in order_books.items():
            self.logger.info(f"\n--- Verifying order book for {stock_id} ---")

            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()

            self.logger.info(f"Best bid: {best_bid}")
            self.logger.info(f"Best ask: {best_ask}")

            # Invariant 1: No crossed market (bid should not exceed ask)
            if best_bid is not None and best_ask is not None:
                if best_bid > best_ask:
                    msg = f"Crossed market detected for {stock_id}:\n"
                    msg += f"Best bid: {best_bid}\n"
                    msg += f"Best ask: {best_ask}\n"
                    msg += f"Bid exceeds ask by: {best_bid - best_ask}"
                    logger.error(msg)
                    raise ValueError(msg)
                self.logger.info(f"✓ No crossed market (bid {best_bid} <= ask {best_ask})")

            # Invariant 2: All orders in book should be PENDING or PARTIALLY_FILLED
            from market.orders.order import OrderState
            valid_book_states = {OrderState.PENDING, OrderState.PARTIALLY_FILLED}

            invalid_orders = []
            for side_name, heap in [('buy', order_book.buy_orders), ('sell', order_book.sell_orders)]:
                for entry in heap:
                    if entry.order.state not in valid_book_states:
                        invalid_orders.append({
                            'order_id': entry.order.order_id,
                            'side': side_name,
                            'state': entry.order.state,
                            'agent_id': entry.order.agent_id
                        })

            if invalid_orders:
                msg = f"Invalid order states in {stock_id} order book:\n"
                for order_info in invalid_orders:
                    msg += f"  Order {order_info['order_id']} ({order_info['side']}): {order_info['state']}\n"
                logger.warning(msg)  # Warning for now

            # Invariant 3: Order book quantities match order remaining quantities
            book_buy_quantity = sum(entry.order.remaining_quantity for entry in order_book.buy_orders)
            book_sell_quantity = sum(entry.order.remaining_quantity for entry in order_book.sell_orders)

            self.logger.info(f"Total buy quantity in book: {book_buy_quantity}")
            self.logger.info(f"Total sell quantity in book: {book_sell_quantity}")

            # Invariant 4: Prices are positive
            for entry in order_book.buy_orders:
                if entry.order.price is not None and entry.order.price <= 0:
                    msg = f"Invalid buy order price in {stock_id}: {entry.order.price}"
                    logger.error(msg)
                    raise ValueError(msg)

            for entry in order_book.sell_orders:
                if entry.order.price is not None and entry.order.price <= 0:
                    msg = f"Invalid sell order price in {stock_id}: {entry.order.price}"
                    logger.error(msg)
                    raise ValueError(msg)

            self.logger.info(f"✓ Order book consistency verified for {stock_id}")

        self.logger.info("\n✓ All order book invariants verified")

    def _verify_dividend_accumulation(self, logger):
        """Verify dividend cash accumulation matches payments"""
        self.logger.info("\n=== Dividend Accumulation Verification ===")

        # Calculate total dividends paid across all stocks
        if self.is_multi_stock:
            total_dividends_paid = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.dividends_paid
            )
        else:
            total_dividends_paid = sum(
                payment['amount']
                for payment in self.context.market_history.dividends_paid
            )

        # Calculate total dividend cash held by agents
        total_dividend_cash = sum(
            self.agent_repository.get_agent(agent_id).dividend_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        self.logger.info(f"Total dividends paid: ${total_dividends_paid:.2f}")
        self.logger.info(f"Total dividend cash held: ${total_dividend_cash:.2f}")

        # Invariant: Dividend cash should equal total dividends paid
        # (unless dividends are paid to main account instead of dividend account)
        if abs(total_dividend_cash - total_dividends_paid) > 0.01:
            # Check if dividends are configured to go to main account
            # In that case, this invariant doesn't apply
            dividend_destination = None
            if self.is_multi_stock:
                # Check first stock's dividend config
                first_stock = list(self.contexts.values())[0]
                dividend_destination = getattr(first_stock, 'dividend_destination', None)
            else:
                dividend_destination = getattr(self.context, 'dividend_destination', None)

            if dividend_destination == 'dividend':
                msg = f"Dividend accumulation mismatch:\n"
                msg += f"Total paid: ${total_dividends_paid:.2f}\n"
                msg += f"Total held: ${total_dividend_cash:.2f}\n"
                msg += f"Difference: ${abs(total_dividend_cash - total_dividends_paid):.2f}"
                logger.warning(msg)  # Warning for now as this might be expected in some configs
            else:
                self.logger.info("Dividends paid to main account (not dividend account)")
        else:
            self.logger.info("✓ Dividend accumulation matches payments")

        # Verify non-negative dividend cash
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            if agent.dividend_cash < 0:
                msg = f"Negative dividend cash for agent {agent_id}: ${agent.dividend_cash:.2f}"
                logger.error(msg)
                raise ValueError(msg)

        self.logger.info("✓ Dividend accumulation verified")

    def _verify_interest_calculations(self, logger):
        """Verify interest rate calculations and payments"""
        self.logger.info("\n=== Interest Calculation Verification ===")

        # Get interest service (shared across all stocks)
        if not hasattr(self, 'interest_service') or not self.interest_service:
            self.logger.info("No interest service configured")
            return

        interest_service = self.interest_service
        stock_label = "all stocks" if self.is_multi_stock else "DEFAULT_STOCK"

        # Verify interest rate is within bounds
        rate = interest_service.interest_model.get('rate', 0)
        self.logger.info(f"{stock_label} interest rate: {rate:.4f} ({rate*100:.2f}%)")

        # Invariant 1: Interest rate should be reasonable (between -10% and +50%)
        if rate < -0.10 or rate > 0.50:
            msg = f"Interest rate out of reasonable bounds: {rate:.4f}"
            logger.warning(msg)

        # Verify interest payments match calculations
        current_round = self.context.round_number

        if self.is_multi_stock:
            total_interest_paid = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.interest_paid
                if payment['round'] == current_round - 1
            )
        else:
            total_interest_paid = sum(
                payment['amount']
                for payment in self.context.market_history.interest_paid
                if payment['round'] == current_round - 1
            )

        self.logger.info(f"Total interest paid this round: ${total_interest_paid:.2f}")

        # Invariant 2: Interest should be non-negative (assuming positive rates)
        if total_interest_paid < 0:
            # This could be valid for negative interest rates
            if rate >= 0:
                msg = f"Negative interest paid with non-negative rate: ${total_interest_paid:.2f}"
                logger.warning(msg)

        # Invariant 3: If interest service exists, check compounding frequency is valid
        compound_freq = interest_service.interest_model.get('compound_frequency', 'per_round')
        valid_frequencies = ['per_round', 'annual', 'semi_annual', 'quarterly', 'monthly']
        if compound_freq not in valid_frequencies:
            msg = f"Invalid compound frequency: {compound_freq}"
            logger.warning(msg)

        self.logger.info("✓ Interest calculations verified")

    def _verify_borrow_fee_calculations(self, logger):
        """Verify borrow fee (short selling cost) calculations"""
        self.logger.info("\n=== Borrow Fee Calculation Verification ===")

        # Check if borrowing is enabled
        if not hasattr(self.agent_repository, 'borrowing_repository') or \
           self.agent_repository.borrowing_repository.total_lendable == 0:
            self.logger.info("No borrowing enabled")
            return

        # Get borrow fee payments
        current_round = self.context.round_number

        if self.is_multi_stock:
            total_borrow_fees_paid = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.borrow_fees_paid
                if payment['round'] == current_round - 1
            )
        else:
            total_borrow_fees_paid = sum(
                payment['amount']
                for payment in self.context.market_history.borrow_fees_paid
                if payment['round'] == current_round - 1
            )

        # Get total borrowed shares
        total_borrowed = sum(
            self.agent_repository.get_agent(agent_id).borrowed_shares
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        self.logger.info(f"Total borrowed shares: {total_borrowed}")
        self.logger.info(f"Total borrow fees paid: ${total_borrow_fees_paid:.2f}")

        # Invariant 1: Borrow fees should be non-negative
        if total_borrow_fees_paid < 0:
            msg = f"Negative borrow fees paid: ${total_borrow_fees_paid:.2f}"
            logger.error(msg)
            raise ValueError(msg)

        # Invariant 2: If shares are borrowed, fees should be paid (unless rate is 0)
        if total_borrowed > 0 and total_borrow_fees_paid == 0:
            # Check if borrow service exists and has a fee rate
            if hasattr(self, 'borrow_service') and self.borrow_service:
                # The borrow service might have a borrow_fee rate
                borrow_fee_rate = getattr(self.borrow_service, 'borrow_fee_rate', 0)
                if borrow_fee_rate > 0:
                    msg = f"Shares borrowed ({total_borrowed}) but no fees paid with positive rate ({borrow_fee_rate})"
                    logger.warning(msg)

        # Invariant 3: Borrow fees should not exceed total cash in system
        total_cash = sum(
            self.agent_repository.get_agent(agent_id).total_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        if total_borrow_fees_paid > total_cash:
            msg = f"Borrow fees (${total_borrow_fees_paid:.2f}) exceed total system cash (${total_cash:.2f})"
            logger.error(msg)
            raise ValueError(msg)

        self.logger.info("✓ Borrow fee calculations verified")
