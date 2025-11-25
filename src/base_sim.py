from datetime import datetime
import traceback
from market.orders.order_book import OrderBook
from market.orders.order import OrderState
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
from market.state.services.leverage_interest_service import LeverageInterestService
from agents.agent_manager.services.agent_decision_service import AgentDecisionService
from market.orders.order_service_factory import OrderServiceFactory
from services.shared_service_factory import SharedServiceFactory
from market.information.base_information_services import InformationService
from market.state.provider_registry import ProviderRegistry
from services.logging_service import LoggingService
from agents.agent_manager.services.borrowing_repository import BorrowingRepository
from agents.agent_manager.services.cash_lending_repository import CashLendingRepository
from verification.simulation_verifier import SimulationVerifier
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
                 model_open_ai = "gpt-oss-20b",  # Usually set via DEFAULT_PARAMS from .env
                 dividend_params: dict = None,
                 interest_params: dict = None,
                 borrow_params: dict = None,
                 infinite_rounds: bool = False,
                 sim_type: str = "default",
                 stock_configs: dict = None,
                 enable_intra_round_margin_checking: bool = False):
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
        self.enable_intra_round_margin_checking = enable_intra_round_margin_checking
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

        # Initialize borrowing repository/repositories (for short selling)
        if self.is_multi_stock:
            # Multi-stock: Create borrowing repository for each stock (FOR LOOP!)
            self.borrowing_repositories = {}
            for stock_id, config in stock_configs.items():
                lendable = config.get('LENDABLE_SHARES', 0)
                self.borrowing_repositories[stock_id] = BorrowingRepository(
                    total_lendable=lendable,
                    allow_partial_borrows=allow_partial_borrows,
                    logger=LoggingService.get_logger(f'borrowing_{stock_id}')
                )
            # For backwards compatibility, expose first stock's repo
            self.borrowing_repository = list(self.borrowing_repositories.values())[0] if self.borrowing_repositories else None

            # Pass dict of borrowing repositories to AgentRepository
            self.agent_repository = AgentRepository(
                agents,
                logger=LoggingService.get_logger('agent_repository'),
                context=self.context,
                borrowing_repositories=self.borrowing_repositories
            )
        else:
            # Single stock: Original behavior (backwards compatible)
            self.borrowing_repository = BorrowingRepository(
                total_lendable=lendable_shares,
                allow_partial_borrows=allow_partial_borrows,
                logger=LoggingService.get_logger('borrowing')
            )

            self.agent_repository = AgentRepository(
                agents,
                logger=LoggingService.get_logger('agent_repository'),
                context=self.context,
                borrowing_repository=self.borrowing_repository
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

        # NEW: Initialize leverage (cash lending) services
        leverage_params = agent_params.get('leverage_params', {})
        self.leverage_enabled = leverage_params.get('enabled', False)

        if self.leverage_enabled:
            # Cash lending pool for leveraged trading
            cash_lending_pool = leverage_params.get('cash_lending_pool', float('inf'))
            allow_partial_cash_borrows = leverage_params.get('allow_partial_borrows', False)

            self.cash_lending_repo = CashLendingRepository(
                total_lendable_cash=cash_lending_pool,
                allow_partial_borrows=allow_partial_cash_borrows,
                logger=LoggingService.get_logger('cash_lending'),
                context=self.context
            )

            # Leverage interest service
            self.leverage_interest_service = LeverageInterestService(
                annual_interest_rate=leverage_params.get('interest_rate', 0.05)
            )

            # Assign cash_lending_repo to all agents
            for agent in self.agent_repository.get_all_agents():
                agent.cash_lending_repo = self.cash_lending_repo

            self.logger.info(
                f"Leverage trading enabled: pool=${cash_lending_pool if cash_lending_pool != float('inf') else 'unlimited'}, "
                f"interest={leverage_params.get('interest_rate', 0.05):.2%}"
            )
        else:
            self.cash_lending_repo = None
            self.leverage_interest_service = None

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
                    logger=LoggingService.get_logger('order_book'),  # Use generic logger for all stocks
                    trades_logger=LoggingService.get_logger('market'),  # Use generic logger for all stocks
                    trade_execution_service=self.trade_execution_service,
                    order_repository=self.order_repository,
                    order_state_manager=self.order_state_manager,
                    agent_repository=self.agent_repository,
                    context=self.contexts[stock_id],
                    is_multi_stock=True,  # Flag multi-stock mode
                    enable_intra_round_margin_checking=self.enable_intra_round_margin_checking
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
                context=self.context,
                enable_intra_round_margin_checking=self.enable_intra_round_margin_checking
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

        # Initialize verification service
        self.verifier = SimulationVerifier(
            agent_repository=self.agent_repository,
            context=self.context,
            contexts=self.contexts if self.is_multi_stock else None,
            order_repository=self.order_repository,
            order_book=self.order_book,
            order_books=self.order_books if self.is_multi_stock else None,
            borrowing_repository=self.borrowing_repository,
            borrowing_repositories=self.borrowing_repositories if self.is_multi_stock else None,
            dividend_service=self.dividend_service,
            dividend_services=self.dividend_services if self.is_multi_stock else None,
            is_multi_stock=self.is_multi_stock,
            infinite_rounds=self.infinite_rounds,
            agent_params=self.agent_params,
            leverage_enabled=self.leverage_enabled,
            cash_lending_repo=self.cash_lending_repo if self.leverage_enabled else None,
            interest_service=self.interest_service,
            borrow_service=self.borrow_service,
            leverage_interest_service=self.leverage_interest_service if self.leverage_enabled else None
        )

    def execute_round(self, round_number):
        """Execute a single round of trading"""
        # Log initial states
        self._log_round_start(round_number)

        # Store pre-round states for verification
        pre_round_states = self.verifier.store_pre_round_states()

        # 1. UPDATE MARKET AND CONTEXT at the beginning of the round
        market_state = self._phase_update_market(round_number)

        # 2. COLLECT NEW AGENT DECISIONS (ORDERS)
        new_orders = self._phase_collect_decisions(market_state, round_number)

        # 3. EXECUTE TRADES using the matching engine
        market_result, stock_market_results = self._phase_match_orders(new_orders, round_number)

        # Update market depth after matching
        self._update_all_market_depths()

        # 4. RECORD DATA for the round
        last_paid_dividend = self._phase_record_data(
            round_number=round_number,
            market_state=market_state,
            market_result=market_result,
            new_orders=new_orders,
            stock_market_results=stock_market_results if self.is_multi_stock else None
        )

        # 5. FINAL END-OF-ROUND UPDATES (including interest/dividend payments)
        self._phase_end_of_round(
            round_number=round_number,
            market_result=market_result,
            stock_market_results=stock_market_results,
            pre_round_states=pre_round_states,
            last_paid_dividend=last_paid_dividend
        )

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

        # Get leverage parameters with centralized defaults
        leverage_params = agent_params.get('leverage_params', {})

        base_params = {
            'agent_id': agent_id,
            'initial_cash': type_specific_params.get('initial_cash', agent_params['initial_cash']),
            'initial_shares': initial_shares_value,
            'position_limit': type_specific_params.get('position_limit', agent_params['position_limit']),
            'allow_short_selling': type_specific_params.get('allow_short_selling', agent_params['allow_short_selling']),
            # Margin parameters (for short selling)
            'margin_requirement': type_specific_params.get('margin_requirement', agent_params.get('margin_requirement', 0.5)),
            'margin_base': type_specific_params.get('margin_base', agent_params.get('margin_base', 'cash')),
            'logger': LoggingService.get_logger('decisions'),
            'info_signals_logger': LoggingService.get_logger('info_signals'),
            'initial_price': self.initial_price,
            # Leverage parameters (for leveraged long positions)
            'leverage_ratio': type_specific_params.get('leverage_ratio', leverage_params.get('max_leverage_ratio', 1.0)),
            'initial_margin': type_specific_params.get('initial_margin', leverage_params.get('initial_margin', 0.5)),
            'maintenance_margin': type_specific_params.get('maintenance_margin', leverage_params.get('maintenance_margin', 0.25)),
        }

        # Check if it's a deterministic agent
        if agent_type in DETERMINISTIC_AGENTS:
            return DETERMINISTIC_AGENTS[agent_type](**base_params)

        # Set model name for hold_llm agent, or use type-specific model override
        model = "hold_llm" if agent_type == "hold_llm" else type_specific_params.get('model', self.model_open_ai)

        # Extract enabled features from agent_params
        from agents.LLMs.services.schema_features import FeatureRegistry
        enabled_features = FeatureRegistry.extract_features_from_config(agent_params)

        # Create LLM agent with appropriate model and feature configuration
        return LLMAgent(
            **base_params,
            agent_type=agent_type,
            model_open_ai=model,
            enabled_features=enabled_features
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
                    # NOTE: Do NOT add DEFAULT_STOCK in multi-stock mode - only actual stocks exist
                    # Reset committed and borrowed positions for all stocks
                    agent.committed_positions = {stock_id: 0 for stock_id in agent.positions.keys()}
                    agent.borrowed_positions = {stock_id: 0 for stock_id in agent.positions.keys()}
                    # Update initial_shares to be the sum across all stocks for verification
                    agent.initial_shares = sum(agent_params['initial_positions'].values())

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

    def _cancel_all_orders_for_stock(self, stock_id: str):
        """Cancel all orders for a specific stock (used during final redemption)"""
        # Get the correct order book for this stock
        if self.is_multi_stock:
            order_book = self.order_books[stock_id]
        else:
            order_book = self.order_book

        # Get all agents and cancel their orders for this stock
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent_orders = order_book.get_agent_orders(agent_id)

            # Get orders for this stock
            stock_orders = [
                order for order in agent_orders.get('buy', []) + agent_orders.get('sell', [])
                if order.stock_id == stock_id and order.state in [OrderState.ACTIVE, OrderState.PARTIALLY_FILLED, OrderState.PENDING]
            ]

            if stock_orders:
                self.logger.info(f"Cancelling {len(stock_orders)} orders for agent {agent_id} on stock {stock_id}")
                # Use the centralized cancellation handler to release commitments and transition state
                self.order_state_manager.handle_agent_all_orders_cancellation(
                    agent_id=agent_id,
                    orders=stock_orders,
                    message="Final redemption"
                )
                # Remove from order book
                order_book.remove_agent_orders(agent_id)

    def _log_round_start(self, round_number: int):
        """Log initial state at the start of a round"""
        self.logger.warning(f"\n=== Round {round_number} ===")
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "Start of ")
        self.order_book.log_order_book_state(f"Start of Round {round_number}")

    def _log_round_end(self, round_number: int):
        """Log final state at the end of a round"""
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "End of ")
        self.order_book.log_order_book_state(f"End of Round {round_number}")

    def _update_all_market_depths(self):
        """Update market depth for all stocks (handles both single and multi-stock)"""
        if self.is_multi_stock:
            for manager in self.market_state_managers.values():
                manager.update_market_depth()
        else:
            self.market_state_manager.update_market_depth()

    def _phase_update_market(self, round_number: int) -> dict:
        """Phase 1: Update market state and prepare for trading

        Args:
            round_number: Current round number

        Returns:
            dict: Market state for agent decision making
        """
        # Update market depths
        self._update_all_market_depths()

        # Get last volume from data recorder
        if self.is_multi_stock:
            # Multi-stock: Get per-stock volumes from market_data
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
                stock_last_volume = last_stock_volumes[stock_id]
                self.contexts[stock_id].update_public_info(round_number, stock_last_volume)

                # Call manager.update() with skip_distribution to avoid multiple calls
                stock_market_state = manager.update(
                    round_number=round_number,
                    last_volume=stock_last_volume,
                    is_round_end=False,
                    skip_distribution=True  # Skip distribution in multi-stock mode
                )
                # Store this stock's state
                market_state['stocks'][stock_id] = stock_market_state

            # Now distribute information ONCE for all stocks
            # Get any manager's information_service (they all share the same one)
            first_manager = list(self.market_state_managers.values())[0]
            information_service = first_manager.information_service

            # Ensure providers are registered (lazy initialization)
            if not information_service.providers:
                for manager in self.market_state_managers.values():
                    ProviderRegistry.register_providers(
                        information_service=information_service,
                        market_state_manager=manager,
                        dividend_service=manager.dividend_service,
                        interest_service=manager.interest_service,
                        borrow_service=manager.borrow_service,
                        hide_fundamental_price=self.hide_fundamental_price
                    )

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

        return market_state

    def _phase_collect_decisions(self, market_state: dict, round_number: int) -> list:
        """Phase 2: Collect agent decisions and create orders

        Args:
            market_state: Current market state
            round_number: Current round number

        Returns:
            List of new orders from agents
        """
        # Log state before collecting decisions
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "Pre-Decision ")

        # Collect new agent decisions (orders)
        new_orders = self.decision_service.collect_decisions(
            market_state=market_state,
            history=self.data_recorder.history,
            round_number=round_number
        )

        # Update market depth after new orders are placed
        self._update_all_market_depths()

        # Log state after decisions but before matching
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "Post-Decision ")
        self.order_book.log_order_book_state(f"After New Orders Round {round_number}")

        return new_orders

    def _phase_match_orders(self, new_orders: list, round_number: int):
        """Phase 3: Match orders and execute trades

        Args:
            new_orders: List of orders to match
            round_number: Current round number

        Returns:
            tuple: (market_result, stock_market_results)
                - market_result: Aggregated result or single-stock result
                - stock_market_results: Per-stock results dict (multi-stock only, else None)
        """
        if self.is_multi_stock:
            # Multi-stock: Match orders FOR EACH STOCK separately
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

            return market_result, stock_market_results
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

            return market_result, None

    def _phase_end_of_round(self, round_number: int, market_result,
                           stock_market_results: dict, pre_round_states: dict,
                           last_paid_dividend: float):
        """Phase 5: End-of-round updates including dividends, redemptions, and interest

        Args:
            round_number: Current round number
            market_result: Result from matching engine
            stock_market_results: Per-stock results (multi-stock only)
            pre_round_states: Pre-round states for verification
            last_paid_dividend: Last paid dividend from phase 4 (for single-stock logging)
        """
        # Check if this is final redemption round
        is_final_round = round_number == self.context._num_rounds - 1 and not self.infinite_rounds

        if self.is_multi_stock:
            # Multi-stock: Let each manager handle end-of-round processing
            total_payments = 0.0

            for stock_id, manager in self.market_state_managers.items():
                # If final round with redemption, cancel all orders for this stock BEFORE redemption
                if is_final_round and self.contexts[stock_id].redemption_value is not None:
                    self.logger.info(f"Final round: cancelling all orders for {stock_id} before redemption")
                    self._cancel_all_orders_for_stock(stock_id)

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
                if manager.dividend_service and manager.dividend_service.dividend_history:
                    stock_payment = manager.dividend_service.dividend_history[-1]
                    total_payments += stock_payment
                    self.logger.info(f"  {stock_id} dividend payment: ${stock_payment:.2f}")

            last_paid_dividend = total_payments
            self.logger.info(f"Total payments across all stocks: ${total_payments:.2f}")
        else:
            # Single stock: Original behavior
            # If final round with redemption, cancel all orders BEFORE redemption
            if is_final_round and self.context.redemption_value is not None:
                stock_id = self.dividend_service.stock_id if self.dividend_service else "DEFAULT_STOCK"
                self.logger.info(f"Final round: cancelling all orders for {stock_id} before redemption")
                self._cancel_all_orders_for_stock(stock_id)

            self.market_state_manager.update(
                round_number=round_number,
                last_volume=market_result.volume,
                is_round_end=True
            )
            self.market_state_manager.update_market_depth()

            # Use last_paid_dividend passed from phase_record_data

        self.logger.info(f"Dividends paid last round: {last_paid_dividend}")

        # Charge interest on borrowed cash for leverage (after market state updates)
        if self.leverage_enabled and self.leverage_interest_service:
            interest_charged = self.leverage_interest_service.charge_interest(
                self.agent_repository.get_all_agents(),
                rounds_per_year=252  # Daily trading
            )
            if interest_charged:
                total_leverage_interest = sum(interest_charged.values())
                # Record leverage interest in market history (use round_number param, not self.context.round_number which was already incremented)
                self.context.record_leverage_interest_charged(
                    amount=total_leverage_interest,
                    round_number=round_number
                )
                self.logger.info(
                    f"Leverage interest charged this round: ${total_leverage_interest:.2f} "
                    f"({len(interest_charged)} agents)"
                )

        # Verify final states
        self.verifier.verify_round_end_states(pre_round_states)

    def _phase_record_data(self, round_number: int, market_state: dict, market_result,
                          new_orders: list, stock_market_results: dict = None):
        """Phase 4: Record round data including dividends and trades

        Args:
            round_number: Current round number
            market_state: Current market state
            market_result: Result from matching engine
            new_orders: List of orders placed this round
            stock_market_results: Per-stock results (multi-stock only)
        """
        # Log state after matching
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "Post-Matching ")
        self.order_book.log_order_book_state(f"After Trades Matched Round {round_number}")

        if round_number == self.context._num_rounds - 1 and not self.infinite_rounds:
            self.logger.info(f"Last round, redeeming shares for fundamental value: {self.context.fundamental_price}")
            self.logger.info(f"Shares are worthless after redemption")

        # Get last paid dividend
        dividends_by_stock = None
        if self.is_multi_stock:
            # Multi-stock: Aggregate dividends from all stocks
            last_paid_dividend = 0.0
            dividends_by_stock = {}  # Track per-stock for detailed recording

            for stock_id, manager in self.market_state_managers.items():
                if manager.dividend_service and manager.dividend_service.dividend_history:
                    # Get last paid dividend for this stock
                    stock_dividend = manager.dividend_service.dividend_history[-1]
                    dividends_by_stock[stock_id] = stock_dividend
                    last_paid_dividend += stock_dividend
                    self.logger.debug(f"Stock {stock_id} last dividend: ${stock_dividend:.2f}")
                else:
                    dividends_by_stock[stock_id] = 0.0

            self.logger.debug(f"Total last paid dividend across all stocks: ${last_paid_dividend:.2f}")

            # Invariance check: Verify aggregation is correct
            expected_total = sum(dividends_by_stock.values())
            if abs(last_paid_dividend - expected_total) > 1e-6:  # Allow for floating point error
                raise ValueError(
                    f"Dividend aggregation invariance violated: "
                    f"last_paid_dividend={last_paid_dividend:.6f} != "
                    f"sum(dividends_by_stock)={expected_total:.6f}"
                )

            # Record per-stock dividends for analytics
            if dividends_by_stock:
                self.data_recorder.record_multi_stock_dividends(
                    round_number=round_number,
                    dividends_by_stock=dividends_by_stock
                )
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
            dividends=last_paid_dividend,
            dividends_by_stock=dividends_by_stock
        )

        # Log final states
        self._log_round_end(round_number)

        return last_paid_dividend
