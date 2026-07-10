from datetime import datetime
import traceback
from market.orders.order_book import OrderBook
from market.orders.order import OrderState
from agents.agent_types import *
from agents.agents_api import *
from market.engine.match_engine import MatchingEngine
from market.engine.market_result import MarketResult
from market.state.market_state_manager import MarketStateManager
from agents.agent_manager.base_agent_manager import AgentManager
from market.data_recorder import DataRecorder
# agents.registry also mirrors deterministic types into AGENT_TYPES on import.
# The validators are re-exported here for backwards compatibility (tests and
# sweep tooling import them from base_sim).
from agents.registry import (
    DETERMINISTIC_AGENTS,
    validate_agent_composition,
    validate_system_prompt_overrides,
)
from agents.agent_factory import AgentFactory
from market.state.sim_context import SimulationContext
from market.stock_market import StockMarket, MarketCollection, DEFAULT_STOCK_ID
from market.orders.order_repository import OrderRepository
from market.state.services.dividend_service import DividendService
from agents.agent_manager.agent_repository import AgentRepository
from typing import Dict, Optional
from market.state.services.interest_service import InterestService
from market.state.services.borrow_service import BorrowService
from market.state.services.leverage_interest_service import LeverageInterestService
from agents.agent_manager.services.agent_decision_service import AgentDecisionService
from market.orders.order_service_factory import OrderServiceFactory
from services.shared_service_factory import SharedServiceFactory
from services.messaging_service import MessagingService
from market.information.base_information_services import InformationService
from market.state.provider_registry import ProviderRegistry
from market.state.services.shock_service import generate_dividend_shocks
from services.logging_service import LoggingService
from agents.agent_manager.services.borrowing_repository import BorrowingRepository
from agents.agent_manager.services.cash_lending_repository import CashLendingRepository
from verification.simulation_verifier import SimulationVerifier
from scenarios.base import FundamentalInfoMode
from calculate_fundamental import regime_fundamental_path


class BaseSimulation:
    """
    The core class for running a trading simulation.

    This class orchestrates the entire simulation, including setting up the
    environment, creating agents, running the market rounds, and collecting data.

    Stocks are held in a MarketCollection (self.markets): each StockMarket
    bundles the per-stock context, order book, borrowing pool, dividend
    service, market state manager, and matching engine. A single-stock run
    is simply the N=1 case keyed by DEFAULT_STOCK_ID; construction and the
    round phases iterate the collection either way. Legacy singular/plural
    attributes (context/contexts, order_book/order_books, ...) are exposed
    as read-only views over the collection.

    Attributes:
        num_rounds (int): The total number of rounds to run the simulation.
        agent_params (dict): Parameters for creating agents.
        sim_type (str): The name of the scenario being run.
        run_dir (Path): The directory where simulation data and plots are saved.
        markets (MarketCollection): Per-stock market components.
        agent_repository (AgentRepository): The repository managing all agents.
        data_recorder (DataRecorder): The service for recording simulation data.
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
                 hide_fundamental_price: bool = True,  # DEPRECATED: use fundamental_info_mode
                 fundamental_info_mode: Optional[FundamentalInfoMode] = None,
                 model_open_ai = "gpt-oss-20b",  # Usually set via DEFAULT_PARAMS from .env
                 llm_temperature: float = 0.0,  # Sampling temperature for LLM agents
                 llm_seed: int = 42,  # Deterministic sampling seed for LLM agents
                 llm_max_concurrency: int = 8,  # Concurrent agent decisions (LLM calls) per round; 1 = serial
                 system_prompt_overrides: dict = None,  # agent_type -> replacement system prompt (prompt-family sweeps)
                 dividend_params: dict = None,
                 interest_params: dict = None,
                 borrow_params: dict = None,
                 infinite_rounds: bool = False,
                 sim_type: str = "default",
                 stock_configs: dict = None,
                 enable_intra_round_margin_checking: bool = False,
                 news_enabled: bool = False):
        SharedServiceFactory.reset()
        # Reset per-run LLM token/cost accounting (issue #104). Sweeps call many
        # runs in one process, so this must clear before the first API call.
        from services.usage_tracker import UsageTracker
        UsageTracker.reset()

        # Populated at end of run() from UsageTracker (None if the run made no
        # LLM calls). run_base_sim merges this into the run's metadata.json.
        self.llm_usage_summary = None

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

        # Multi-stock scenarios pass stock_configs; the mode drives the
        # agent-facing market_state shape and recording formats, so it is
        # kept explicit rather than inferred from the stock count.
        self.is_multi_stock = stock_configs is not None
        self.stock_configs = stock_configs

        if self.is_multi_stock and not stock_configs:
            raise ValueError("stock_configs cannot be empty when multi-stock mode is enabled")

        # Normalize to per-stock configs so single-stock is just the N=1 case
        if self.is_multi_stock:
            normalized_configs = stock_configs
        else:
            normalized_configs = {
                DEFAULT_STOCK_ID: {
                    'INITIAL_PRICE': initial_price,
                    'FUNDAMENTAL_PRICE': fundamental_price,
                    'REDEMPTION_VALUE': redemption_value,
                    'TRANSACTION_COST': transaction_cost,
                    'LENDABLE_SHARES': lendable_shares,
                    'DIVIDEND_PARAMS': dividend_params,
                }
            }

        self.markets = MarketCollection()
        for stock_id, config in normalized_configs.items():
            self.markets.add(StockMarket(
                stock_id=stock_id,
                config=config,
                context=SimulationContext(
                    num_rounds=num_rounds,
                    initial_price=config['INITIAL_PRICE'],
                    fundamental_price=config['FUNDAMENTAL_PRICE'],
                    redemption_value=config.get('REDEMPTION_VALUE'),
                    transaction_cost=config.get('TRANSACTION_COST', 0.0),
                    logger=LoggingService.get_logger(self._stock_logger_name('context', stock_id)),
                    infinite_rounds=self.infinite_rounds
                ),
                # Style category for style-level dividend shocks; multi-stock only
                style=self._stock_style_from_config(config) if self.is_multi_stock else None,
            ))

        # Per-stock volume of the previous round, used to build the next
        # round's market state (empty means round 0: no prior volume).
        self._last_round_volumes: Dict[str, float] = {}

        # Basic simulation parameters
        self.fundamental_volatility = fundamental_volatility
        self.news_enabled = news_enabled
        self.model_open_ai = model_open_ai
        self.llm_temperature = llm_temperature
        self.llm_seed = llm_seed

        # System-prompt overrides (prompt-family robustness sweeps, issue #102):
        # agent_type -> replacement system prompt. Validated up front so a typo'd
        # agent type fails at construction, not silently mid-sweep.
        self.system_prompt_overrides = system_prompt_overrides or {}
        validate_system_prompt_overrides(self.system_prompt_overrides)

        # Fundamental info mode: controls what agents see
        # Handle backwards compatibility with hide_fundamental_price
        if fundamental_info_mode is not None:
            self.fundamental_info_mode = fundamental_info_mode
        elif hide_fundamental_price:
            self.fundamental_info_mode = FundamentalInfoMode.PROCESS_ONLY
        else:
            self.fundamental_info_mode = FundamentalInfoMode.FULL

        # Keep hide_fundamental_price for components that still use it
        self.hide_fundamental_price = self.fundamental_info_mode != FundamentalInfoMode.FULL

        # Set default agent parameters if none provided
        self.agent_params = agent_params

        # Configure the social-feed transform (issue #95): readers see the
        # transformed feed, raw messages are still logged. Seeded with
        # llm_seed so 'scrambled' reproduces exactly on matched-seed runs.
        MessagingService.configure(
            transform=(agent_params or {}).get('FEED_TRANSFORM', 'identity'),
            seed=llm_seed,
        )

        # Round-lifecycle hooks: callables invoked as hook(sim, round_number)
        # before/after every round. Lets extensions attach behavior without
        # editing the orchestrator (register via register_before_round /
        # register_after_round).
        self._before_round_hooks = []
        self._after_round_hooks = []

        # Initialize agents with explicit parameters (construction is
        # delegated to AgentFactory; see agents/agent_factory.py)
        self.agent_factory = AgentFactory(
            initial_price=initial_price,
            model_open_ai=model_open_ai,
            llm_temperature=llm_temperature,
            llm_seed=llm_seed,
            fundamental_info_mode=self.fundamental_info_mode,
            system_prompt_overrides=self.system_prompt_overrides,
            logger=self.logger,
        )
        agents = self.agent_factory.initialize_agents(self.agent_params)

        # Get borrow configuration
        borrow_model = self.agent_params.get('borrow_model', {})
        allow_partial_borrows = borrow_model.get('allow_partial_borrows', True)

        # Initialize a borrowing repository per stock (for short selling)
        for stock_id, market in self.markets.items():
            market.borrowing_repository = BorrowingRepository(
                total_lendable=market.config.get('LENDABLE_SHARES', 0),
                allow_partial_borrows=allow_partial_borrows,
                logger=LoggingService.get_logger(self._stock_logger_name('borrowing', stock_id))
            )

        # AgentRepository keys per-stock borrow lookups off which parameter
        # is passed, so the mode picks the constructor form.
        if self.is_multi_stock:
            self.agent_repository = AgentRepository(
                agents,
                logger=LoggingService.get_logger('agent_repository'),
                context=self.context,
                borrowing_repositories=self.markets.borrowing_repositories()
            )
        else:
            self.agent_repository = AgentRepository(
                agents,
                logger=LoggingService.get_logger('agent_repository'),
                context=self.context,
                borrowing_repository=self.markets.primary.borrowing_repository
            )

        # Order book per stock
        for stock_id, market in self.markets.items():
            market.order_book = OrderBook(
                context=market.context,
                logger=LoggingService.get_logger(self._stock_logger_name('order_book', stock_id)),
                order_repository=self.order_repository
            )

        # Initialize shared services (the factory's multi-stock mode changes
        # commitment lookups, so it must follow the scenario mode exactly)
        if self.is_multi_stock:
            SharedServiceFactory.initialize(
                order_books=self.markets.order_books(),
                transaction_cost={
                    stock_id: market.config.get('TRANSACTION_COST', 0.0)
                    for stock_id, market in self.markets.items()
                }
            )
        else:
            SharedServiceFactory.initialize(
                order_book=self.order_book,
                transaction_cost=transaction_cost
            )

        # Create order services through factory
        self.order_state_manager, self.trade_execution_service = OrderServiceFactory.create_services(
            order_repository=self.order_repository,
            agent_repository=self.agent_repository,
            order_book= self.order_book,
            logger=LoggingService.get_logger('order_state')
        )

        # Dividend service per stock that pays dividends
        for stock_id, market in self.markets.items():
            stock_dividend_params = market.config.get('DIVIDEND_PARAMS')
            if stock_dividend_params:
                market.dividend_service = DividendService(
                    agent_repository=self.agent_repository,
                    logger=LoggingService.get_logger(
                        f'dividend_{stock_id}' if self.is_multi_stock else 'market_state'),
                    dividend_params=stock_dividend_params,
                    redemption_value=market.context.redemption_value,
                    stock_id=stock_id
                )

        # Piecewise fundamental path for dividend regime schedules (issue #96):
        # None for stationary scenarios, one value per round otherwise
        for market in self.markets:
            market.fundamental_path = self._build_fundamental_path(market, interest_params)
            if market.fundamental_path is not None:
                market.context.fundamental_price = market.fundamental_path[0]

        # Initialize dividend shock structure (for systematic vs idiosyncratic shocks)
        # Can be configured at the agent_params level or individually per stock
        self.shock_config = agent_params.get('shock_structure', {}) if agent_params else {}
        self.shock_enabled = self.shock_config.get('enabled', False)
        self._current_round_shocks = None  # Stores shocks for current round (for data recording)

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

            # Leverage interest service (same rate as cash interest for consistency)
            self.leverage_interest_service = LeverageInterestService(
                interest_rate=leverage_params.get('interest_rate', 0.05)
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

        # Market state manager per stock; the shared information service is
        # created afterwards and attached to every manager
        for stock_id, market in self.markets.items():
            market.market_state_manager = MarketStateManager(
                context=market.context,
                order_book=market.order_book,
                agent_repository=self.agent_repository,
                logger=LoggingService.get_logger(
                    f'market_state_{stock_id}' if self.is_multi_stock else 'market_state'),
                information_service=None,  # Set below once the shared service exists
                dividend_service=market.dividend_service,
                interest_service=self.interest_service,
                borrow_service=self.borrow_service,
                hide_fundamental_price=self.hide_fundamental_price,
                news_enabled=self.news_enabled
            )

        # A managers dict switches the information service into multi-stock
        # signal generation, so it is only passed in multi-stock mode
        self.information_service = InformationService(
            agent_repository=self.agent_repository,
            market_state_managers=self.markets.market_state_managers() if self.is_multi_stock else None,
            info_capabilities_config=self.agent_params.get('info_capabilities')
        )
        for market in self.markets:
            market.market_state_manager.information_service = self.information_service

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
            order_repository=self.order_repository,
            position_calculator=SharedServiceFactory.get_position_calculator(),
            commitment_calculator=SharedServiceFactory.get_commitment_calculator()
        )

        # Matching engine per stock
        for stock_id, market in self.markets.items():
            market.matching_engine = MatchingEngine(
                order_book=market.order_book,
                agent_manager=self.agent_manager,
                logger=LoggingService.get_logger('order_book'),
                trades_logger=LoggingService.get_logger('market' if self.is_multi_stock else 'trades'),
                trade_execution_service=self.trade_execution_service,
                order_repository=self.order_repository,
                order_state_manager=self.order_state_manager,
                agent_repository=self.agent_repository,
                context=market.context,
                is_multi_stock=self.is_multi_stock,
                enable_intra_round_margin_checking=self.enable_intra_round_margin_checking,
                stock_id=stock_id
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
            context=self.context,
            max_concurrency=llm_max_concurrency
        )

        # Initialize verification service
        self.verifier = SimulationVerifier(
            agent_repository=self.agent_repository,
            context=self.context,
            contexts=self.markets.contexts() if self.is_multi_stock else None,
            order_repository=self.order_repository,
            order_book=self.order_book,
            order_books=self.markets.order_books() if self.is_multi_stock else None,
            borrowing_repository=self.borrowing_repository,
            borrowing_repositories=self.markets.borrowing_repositories() if self.is_multi_stock else None,
            dividend_service=self.dividend_service,
            dividend_services=self.markets.dividend_services() if self.is_multi_stock else None,
            is_multi_stock=self.is_multi_stock,
            infinite_rounds=self.infinite_rounds,
            agent_params=self.agent_params,
            leverage_enabled=self.leverage_enabled,
            cash_lending_repo=self.cash_lending_repo if self.leverage_enabled else None,
            interest_service=self.interest_service,
            borrow_service=self.borrow_service,
            leverage_interest_service=self.leverage_interest_service if self.leverage_enabled else None,
            fundamental_paths={
                stock_id: market.fundamental_path
                for stock_id, market in self.markets.items()
            }
        )

    # ------------------------------------------------------------------
    # Legacy singular/plural component views (kept for compatibility with
    # external consumers; internally the collection is the source of truth)
    # ------------------------------------------------------------------

    @property
    def context(self):
        """The first stock's context (the only one in single-stock mode)."""
        return self.markets.primary.context

    @property
    def contexts(self):
        return self.markets.contexts()

    @property
    def order_book(self):
        return self.markets.primary.order_book

    @property
    def order_books(self):
        return self.markets.order_books()

    @property
    def market_state_manager(self):
        return self.markets.primary.market_state_manager

    @property
    def market_state_managers(self):
        return self.markets.market_state_managers()

    @property
    def matching_engine(self):
        return self.markets.primary.matching_engine

    @property
    def matching_engines(self):
        return {stock_id: market.matching_engine for stock_id, market in self.markets.items()}

    @property
    def dividend_service(self):
        """First stock that pays dividends, or None (legacy behavior)."""
        return next((m.dividend_service for m in self.markets if m.dividend_service), None)

    @property
    def dividend_services(self):
        return self.markets.dividend_services()

    @property
    def borrowing_repository(self):
        return self.markets.primary.borrowing_repository

    @property
    def borrowing_repositories(self):
        return self.markets.borrowing_repositories()

    @property
    def fundamental_path(self):
        return self.markets.primary.fundamental_path

    def _stock_logger_name(self, base: str, stock_id: str) -> str:
        """Legacy logger naming: bare names in single-stock mode, per-stock
        suffixes in multi-stock mode."""
        return f"{base}_{stock_id}" if self.is_multi_stock else base

    @staticmethod
    def _stock_style_from_config(config: dict) -> Optional[str]:
        """Style category for shock lookup: stock level, falling back to
        DIVIDEND_PARAMS."""
        style = config.get('style')
        if style is None:
            style = (config.get('DIVIDEND_PARAMS') or {}).get('style')
        return style

    def register_before_round(self, hook):
        """Register a callable invoked as hook(sim, round_number) before each round."""
        self._before_round_hooks.append(hook)

    def register_after_round(self, hook):
        """Register a callable invoked as hook(sim, round_number) after each round."""
        self._after_round_hooks.append(hook)

    def execute_round(self, round_number):
        """Execute a single round of trading"""
        for hook in self._before_round_hooks:
            hook(self, round_number)

        # Log initial states
        self._log_round_start(round_number)

        # Reset margin call costs and transaction fees for this round (for verification tracking)
        for agent in self.agent_repository.get_all_agents():
            agent.margin_call_cost_this_round = 0.0
            agent.transaction_fees_this_round = 0.0

        # Store pre-round states for verification
        pre_round_states = self.verifier.store_pre_round_states()

        # 1. UPDATE MARKET AND CONTEXT at the beginning of the round
        market_state = self._phase_update_market(round_number)

        # 2. COLLECT NEW AGENT DECISIONS (ORDERS)
        new_orders = self._phase_collect_decisions(market_state, round_number)

        # 3. EXECUTE TRADES using the matching engine
        market_result, results_by_stock = self._phase_match_orders(new_orders, round_number)

        # Update market depth after matching
        self._update_all_market_depths()

        # 4. RECORD DATA for the round
        last_paid_dividend = self._phase_record_data(
            round_number=round_number,
            market_state=market_state,
            market_result=market_result,
            new_orders=new_orders
        )

        # 5. FINAL END-OF-ROUND UPDATES (including interest/dividend payments)
        self._phase_end_of_round(
            round_number=round_number,
            results_by_stock=results_by_stock,
            pre_round_states=pre_round_states,
            last_paid_dividend=last_paid_dividend
        )

        for hook in self._after_round_hooks:
            hook(self, round_number)

    def create_agent(self, agent_id: int, agent_type: str, agent_params: dict):
        """Create one agent (delegates to AgentFactory; kept for compatibility)."""
        return self.agent_factory.create_agent(agent_id, agent_type, agent_params)

    def initialize_agents(self, agent_params: dict):
        """Build the agent population (delegates to AgentFactory; kept for compatibility)."""
        return self.agent_factory.initialize_agents(agent_params)


    def run(self):
        """Base simulation run logic"""
        try:
            # Clear any cached news from previous simulations (if running multiple in same process)
            if self.news_enabled:
                from market.information.information_providers import NewsProvider
                NewsProvider._multi_stock_cache.clear()

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
            # Persist realized LLM token/cost accounting (issue #104). Best-effort:
            # a failure here must not mask a simulation result.
            try:
                self._save_llm_usage()
            except Exception as e:
                LoggingService.log_simulation(f"Failed to save LLM usage: {str(e)}")
       # Clean up expired orders at end of round

    def _save_llm_usage(self):
        """Write data/llm_usage.csv and stash the run's usage summary (issue #104).

        No CSV is written for all-deterministic runs (no LLM calls). The summary
        is left on self.llm_usage_summary for run_base_sim to fold into metadata.
        """
        from services.usage_tracker import UsageTracker
        csv_path = self.data_dir / 'llm_usage.csv'
        UsageTracker.save_csv(csv_path)
        summary = UsageTracker.summary()
        self.llm_usage_summary = summary if summary.get("calls", 0) else None

    def _generate_dividend_shocks(self) -> dict:
        """Generate systematic and style-level dividend shocks for the current
        round (see market/state/services/shock_service.py)."""
        return generate_dividend_shocks(self.shock_config, self.shock_enabled, self.logger)

    def _cancel_all_orders_for_stock(self, stock_id: str):
        """Cancel all orders for a specific stock (used during final redemption)"""
        order_book = self.markets[stock_id].order_book

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
        """Update market depth for every stock"""
        for market in self.markets:
            market.market_state_manager.update_market_depth()

    def _build_fundamental_path(self, market: StockMarket, interest_params: dict) -> Optional[list]:
        """Build the per-round fundamental value path for a dividend regime
        schedule (issue #96). Returns None for stationary scenarios.

        The path follows the no-arbitrage recursion V_t = (e_t + V_{t+1})/(1+r)
        anchored at the redemption value (finite horizon) or the terminal-regime
        continuation value E[d]/r (infinite horizon). It is applied to the
        stock's fundamental_price at the start of every round, so recorded
        market data reflects the active regime.
        """
        dividend_params = market.config.get('DIVIDEND_PARAMS')
        if not dividend_params or not dividend_params.get('regime_schedule'):
            return None

        interest_rate = (
            interest_params or self.agent_params.get('interest_model', {})
        ).get('rate', 0.05)
        num_rounds = market.context._num_rounds

        # Terminal anchor: redemption value for finite horizons, terminal-regime
        # continuation value E[d]/r otherwise (None -> computed by the helper)
        if not self.infinite_rounds and market.context.redemption_value is not None:
            terminal_anchor = market.context.redemption_value
        else:
            terminal_anchor = None

        path, terminal_value = regime_fundamental_path(
            dividend_params, num_rounds, interest_rate, terminal_anchor
        )
        self.logger.info(
            f"Dividend regime schedule active for {market.stock_id}: "
            f"fundamental path {path[0]:.4f} -> {path[-1]:.4f}, "
            f"terminal anchor {terminal_value:.4f}"
        )
        return path

    def _ensure_providers_registered(self):
        """Register information providers once, on the first distributing
        round (matches the legacy lazy registration inside
        MarketStateManager._distribute_market_information)."""
        if self.information_service.providers:
            return
        for market in self.markets:
            manager = market.market_state_manager
            ProviderRegistry.register_providers(
                information_service=self.information_service,
                market_state_manager=manager,
                dividend_service=manager.dividend_service,
                interest_service=manager.interest_service,
                borrow_service=manager.borrow_service,
                hide_fundamental_price=self.hide_fundamental_price,
                news_enabled=self.news_enabled,
                total_rounds=self.context._num_rounds
            )

    def _phase_update_market(self, round_number: int) -> dict:
        """Phase 1: Update market state and prepare for trading

        Args:
            round_number: Current round number

        Returns:
            dict: Market state for agent decision making (flat single-stock
            shape, or {'stocks': {...}, 'is_multi_stock': True} in
            multi-stock mode)
        """
        # Update market depths
        self._update_all_market_depths()

        # Update each stock and collect its state; information distribution
        # happens once afterwards, for all stocks
        per_stock_state = {}
        for stock_id, market in self.markets.items():
            last_volume = self._last_round_volumes.get(stock_id, 0)

            if market.fundamental_path is not None:
                # Regime schedules: fundamental reflects the active dividend regime
                market.context.fundamental_price = market.fundamental_path[
                    min(round_number, len(market.fundamental_path) - 1)
                ]
            market.context.update_public_info(round_number, last_volume)
            per_stock_state[stock_id] = market.market_state_manager.update(
                round_number=round_number,
                last_volume=last_volume,
                is_round_end=False,
                skip_distribution=True
            )

        # Ensure providers are registered (lazy initialization)
        self._ensure_providers_registered()

        # Generate news for all stocks in ONE LLM call (multi-stock only;
        # single-stock news flows through the provider during distribution)
        if self.news_enabled and self.is_multi_stock:
            from market.information.information_types import InformationType
            news_provider = self.information_service.providers.get(InformationType.NEWS)
            if news_provider:
                news_provider.generate_news_for_all_stocks(
                    round_number=round_number,
                    managers=self.markets.market_state_managers()
                )

        # Distribute information ONCE for all stocks
        self.information_service.distribute_information(round_number)

        if self.is_multi_stock:
            return {
                'stocks': per_stock_state,
                'round_number': round_number + 1,
                'is_multi_stock': True
            }
        return per_stock_state[DEFAULT_STOCK_ID]

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
            tuple: (market_result, results_by_stock)
                - market_result: Single-stock result, or aggregated result in
                  multi-stock mode (trades/volume summed over stocks)
                - results_by_stock: Per-stock results dict
        """
        results_by_stock = {}
        for stock_id, market in self.markets.items():
            if self.is_multi_stock:
                # Route each order to its own stock's engine
                stock_orders = [o for o in new_orders if o.stock_id == stock_id]
                self.logger.info(f"=== Matching {len(stock_orders)} orders for {stock_id} ===")
            else:
                stock_orders = new_orders

            result = market.matching_engine.match_orders(
                stock_orders,
                market.context.current_price,
                round_number + 1
            )
            results_by_stock[stock_id] = result

            # Update price and round number for this stock
            market.context.current_price = result.price
            market.context.round_number = round_number + 1

        # Remember per-stock volume for the next round's market state
        self._last_round_volumes = {
            stock_id: result.volume for stock_id, result in results_by_stock.items()
        }

        if self.is_multi_stock:
            # Aggregate trades and volume across stocks for consumers that
            # expect a single result (recorder); per-stock prices live in
            # each stock's context
            all_trades = [t for result in results_by_stock.values() for t in result.trades]
            market_result = MarketResult(
                price=results_by_stock[self.markets.primary.stock_id].price,
                trades=all_trades,
                volume=sum(result.volume for result in results_by_stock.values())
            )

            # Update agent wealth with all stock prices; in single-stock mode
            # the matching engine already did this with its scalar price
            self.agent_repository.update_all_wealth(self.markets.prices())

            return market_result, results_by_stock

        return results_by_stock[DEFAULT_STOCK_ID], results_by_stock

    def _phase_end_of_round(self, round_number: int, results_by_stock: dict,
                           pre_round_states: dict, last_paid_dividend: float):
        """Phase 5: End-of-round updates including dividends, redemptions, and interest

        Args:
            round_number: Current round number
            results_by_stock: Per-stock matching results
            pre_round_states: Pre-round states for verification
            last_paid_dividend: Last paid dividend from phase 4 (for single-stock logging)
        """
        # Check if this is final redemption round
        is_final_round = round_number == self.context._num_rounds - 1 and not self.infinite_rounds

        # Generate dividend shocks ONCE for this round (before processing any stocks)
        # This ensures systematic shock is the same for all stocks
        shocks = self._generate_dividend_shocks()
        self._current_round_shocks = shocks  # Store for data recording

        total_payments = 0.0
        for stock_id, market in self.markets.items():
            # If final round with redemption, cancel all orders for this stock BEFORE redemption
            if is_final_round and market.context.redemption_value is not None:
                self.logger.info(f"Final round: cancelling all orders for {stock_id} before redemption")
                self._cancel_all_orders_for_stock(stock_id)

            # Style-level shock for this stock (single-stock has no style)
            style_shock = shocks['styles'].get(market.style, 0.0) if market.style else 0.0

            # Manager.update() with is_round_end=True handles all end-of-round
            # processing; pass THIS stock's volume, not the aggregated total
            market.market_state_manager.update(
                round_number=round_number,
                last_volume=results_by_stock[stock_id].volume,
                is_round_end=True,
                systematic_shock=shocks['systematic'],
                style_shock=style_shock
            )
            market.market_state_manager.update_market_depth()

            # Aggregate total payments from all stocks for logging
            # dividend_history contains DividendRealization objects
            if self.is_multi_stock and market.dividend_service and market.dividend_service.dividend_history:
                last_realization = market.dividend_service.dividend_history[-1]
                stock_payment = last_realization.total_dividend
                total_payments += stock_payment
                self.logger.info(f"  {stock_id} dividend payment: ${stock_payment:.2f}")

        if self.is_multi_stock:
            last_paid_dividend = total_payments
            self.logger.info(f"Total payments across all stocks: ${total_payments:.2f}")

        self.logger.info(f"Dividends paid last round: {last_paid_dividend}")

        # Charge interest on borrowed cash for leverage (after market state updates)
        if self.leverage_enabled and self.leverage_interest_service:
            interest_charged = self.leverage_interest_service.charge_interest(
                self.agent_repository.get_all_agents()
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
                          new_orders: list):
        """Phase 4: Record round data including dividends and trades

        Args:
            round_number: Current round number
            market_state: Current market state
            market_result: Result from matching engine
            new_orders: List of orders placed this round
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
            realizations_by_stock = {}  # Track full realizations for shock logging

            for stock_id, market in self.markets.items():
                if market.dividend_service and market.dividend_service.dividend_history:
                    # Get last paid dividend for this stock
                    # dividend_history contains DividendRealization objects
                    last_realization = market.dividend_service.dividend_history[-1]
                    stock_dividend = last_realization.total_dividend
                    dividends_by_stock[stock_id] = stock_dividend
                    realizations_by_stock[stock_id] = last_realization
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

            # Record per-stock dividends for analytics (including shock breakdowns)
            if dividends_by_stock:
                self.data_recorder.record_multi_stock_dividends(
                    round_number=round_number,
                    dividends_by_stock=dividends_by_stock,
                    realizations_by_stock=realizations_by_stock
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
