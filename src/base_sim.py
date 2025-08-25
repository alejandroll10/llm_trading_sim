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
                 sim_type: str = "default"):

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

        # Create context with logger
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
        self.agent_repository = AgentRepository(
            agents,
            logger=LoggingService.get_logger('agent_repository'),
            context=self.context,
            borrowing_repository=BorrowingRepository(
                total_lendable=lendable_shares,
                logger=LoggingService.get_logger('borrowing')
            )
        )
        # Initialize components in correct order
        self.order_book = OrderBook(
            context=self.context,
            logger=LoggingService.get_logger('order_book'),
            order_repository=self.order_repository
        )
        
        # Initialize shared services
        SharedServiceFactory.initialize(self.order_book)
        
        # Create order services through factory
        self.order_state_manager, self.trade_execution_service = OrderServiceFactory.create_services(
            order_repository=self.order_repository,
            agent_repository=self.agent_repository,
            order_book= self.order_book,
            logger=LoggingService.get_logger('order_state')
        )
        
        # Initialize dividend service first
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
        # Create and set information service
        information_service = InformationService(agent_repository=self.agent_repository)
        # Create market state manager with both services
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

        # Initialize matching engine last
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
        self.market_state_manager.update_market_depth()
        
        # Get last volume from data recorder
        last_volume = (
            self.data_recorder.history[-1]['total_volume'] 
            if self.data_recorder.history 
            else 0
        )
        self.context.update_public_info(round_number, last_volume)
        
        # Update market components (e.g., fundamental price, dividends if any)
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
        
        self.market_state_manager.update_market_depth()
        
        # Log state after decisions but before matching
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "Post-Decision ")
        self.order_book.log_order_book_state(f"After New Orders Round {round_number}")
        
        # 3. EXECUTE TRADES using the matching engine
        market_result = self.matching_engine.match_orders(
            new_orders,
            self.context.current_price,
            round_number + 1
        )
        self.market_state_manager.update_market_depth()
        
        # Log state after matching
        LoggingService.log_all_agent_states(self.agent_repository, round_number, "Post-Matching ")
        self.order_book.log_order_book_state(f"After Trades Matched Round {round_number}")
        
        # Update price and round number in the context
        self.context.current_price = market_result.price
        self.context.round_number = round_number + 1

        if round_number == self.context._num_rounds - 1 and not self.infinite_rounds:
            self.logger.info(f"Last round, redeeming shares for fundamental value: {self.context.fundamental_price}")
            self.logger.info(f"Shares are worthless after redemption")
        
        # 4. RECORD DATA for the round
        
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

        base_params = {
            'agent_id': agent_id,
            'initial_cash': type_specific_params.get('initial_cash', agent_params['initial_cash']),
            'initial_shares': type_specific_params.get('initial_shares', agent_params['initial_shares']),
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
                agents.append(self.create_agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    agent_params=agent_params
                ))
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
        
        self.logger.info(f"Round payments - Dividends: ${dividend_payment:.2f}, Interest: ${interest_payment:.2f}")
        
        # Verify round-by-round changes
        cash_difference = total_cash_post - total_cash_pre
        total_round_payments = dividend_payment + interest_payment
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
        total_historical_dividends = sum(payment['amount'] for payment in self.context.market_history.dividends_paid)
        total_historical_interest = sum(payment['amount'] for payment in self.context.market_history.interest_paid)
        expected_total_cash = initial_cash + total_historical_dividends + total_historical_interest
        
        self.logger.info(f"\n=== System-wide cash verification ===")
        self.logger.info(f"Initial cash: ${initial_cash:.2f}")
        self.logger.info(f"Total historical dividends: ${total_historical_dividends:.2f}")
        self.logger.info(f"Total historical interest: ${total_historical_interest:.2f}")
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
        self.logger.info(f"Per agent: {self.agent_params['initial_shares']}")
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
            self.logger.info("Final redemption: All shares successfully redeemed")
        elif total_shares_current != initial_shares:
            msg = (f"Total shares in system changed from initial allocation:\n"
                   f"Initial shares: {initial_shares}\n"
                   f"Current shares: {total_shares_current}\n"
                   f"Share changes this round:")
            for agent_id in pre_shares:
                change = post_shares[agent_id] - pre_shares[agent_id]
                msg += f"\nAgent {agent_id}: {pre_shares[agent_id]} -> {post_shares[agent_id]} (Δ{change})"
            logger.error(msg)
            raise ValueError(msg)

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
