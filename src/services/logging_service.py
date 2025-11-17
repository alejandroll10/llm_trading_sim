"""Refactored LoggingService using modular logging components."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from logging_utils.logger_factory import LoggerFactory
from logging_utils.csv_header_manager import CSVHeaders, CSVHeaderManager
from logging_utils.csv_logger import CSVLogger
from services.logging_models import LogFormatter, LogMessage, AgentStateLogEntry

if TYPE_CHECKING:
    from agents.agent_manager.agent_repository import AgentRepository
from market.trade import Trade


class LoggingService:
    """Centralized logging service with singleton pattern."""

    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _run_dir: Optional[Path] = None
    _latest_dir: Optional[Path] = None
    _data_dir: Optional[Path] = None
    _latest_data_dir: Optional[Path] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, run_id: str):
        """Initialize all loggers and directories."""
        # Create base directory structure
        base_log_dir = Path('logs')
        cls._run_dir = base_log_dir / run_id
        cls._latest_dir = base_log_dir / 'latest_sim'
        cls._data_dir = cls._run_dir / 'data'

        # Extract scenario name from run_id (format: scenario_name/timestamp)
        scenario_name = run_id.split('/')[0] if '/' in run_id else run_id

        # Create scenario-specific directory in latest_sim
        scenario_dir = cls._latest_dir / scenario_name
        cls._latest_data_dir = scenario_dir / 'data'

        # Create all necessary directories
        cls._run_dir.mkdir(parents=True, exist_ok=True)
        cls._latest_dir.mkdir(parents=True, exist_ok=True)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        cls._data_dir.mkdir(exist_ok=True)
        cls._latest_data_dir.mkdir(parents=True, exist_ok=True)

        # Setup console handler for warnings and errors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Initialize all CSV files with headers
        cls._initialize_csv_headers()

        # Setup all loggers
        cls._setup_all_loggers(console_handler)

        # Prevent duplicate messages
        for logger in cls._loggers.values():
            logger.propagate = False

    @classmethod
    def _initialize_csv_headers(cls):
        """Initialize all CSV files with headers."""
        # Validation errors CSV
        CSVHeaderManager.initialize_csv_files(
            [cls._run_dir / 'validation_errors.csv', cls._latest_dir / 'validation_errors.csv'],
            CSVHeaders.VALIDATION_ERRORS
        )

        # Margin calls CSV
        CSVHeaderManager.initialize_csv_files(
            [cls._run_dir / 'margin_calls.csv', cls._latest_dir / 'margin_calls.csv'],
            CSVHeaders.MARGIN_CALLS
        )

        # Structured decisions CSV
        CSVHeaderManager.initialize_csv_files(
            [cls._run_dir / 'structured_decisions.csv', cls._latest_dir / 'structured_decisions.csv'],
            CSVHeaders.STRUCTURED_DECISIONS
        )

    @classmethod
    def _setup_all_loggers(cls, console_handler: logging.Handler):
        """Setup all loggers using the factory."""
        # CSV loggers with dual output (run_dir + latest_dir)
        cls._loggers['validation_errors'] = LoggerFactory.create_csv_logger(
            'validation_errors', cls._run_dir, cls._latest_dir,
            'validation_errors.csv', console_handler
        )

        cls._loggers['margin_calls'] = LoggerFactory.create_csv_logger(
            'margin_calls', cls._run_dir, cls._latest_dir,
            'margin_calls.csv', console_handler
        )

        cls._loggers['structured_decisions'] = LoggerFactory.create_csv_logger(
            'structured_decisions', cls._run_dir, cls._latest_dir,
            'structured_decisions.csv', console_handler
        )

        # Regular loggers with dual output
        cls._loggers['market'] = LoggerFactory.create_logger(
            'market', cls._run_dir, cls._latest_dir,
            'market.log', console_handler
        )

        cls._loggers['borrow'] = LoggerFactory.create_logger(
            'borrow', cls._run_dir, cls._latest_dir,
            'borrow.log', console_handler
        )

        cls._loggers['borrowing'] = LoggerFactory.create_logger(
            'borrowing', cls._run_dir, cls._latest_dir,
            'borrowing.log', console_handler
        )

        # Simple loggers (run_dir only, mode='w')
        cls._loggers['decisions'] = LoggerFactory.create_simple_logger(
            'decisions', cls._run_dir, 'decisions.log', console_handler
        )

        cls._loggers['simulation'] = LoggerFactory.create_simple_logger(
            'simulation', cls._run_dir, 'simulation.log', console_handler
        )

        cls._loggers['agents'] = LoggerFactory.create_simple_logger(
            'agents', cls._run_dir, 'agents.log', console_handler
        )

        cls._loggers['order_state'] = LoggerFactory.create_simple_logger(
            'order_state', cls._run_dir, 'order_state.log', console_handler
        )

        cls._loggers['info_signals'] = LoggerFactory.create_simple_logger(
            'info_signals', cls._run_dir, 'info_signals.log', console_handler
        )

        cls._loggers['order_book'] = LoggerFactory.create_logger(
            'order_book', cls._run_dir, None,
            'order_book.log', console_handler, use_dual_output=False
        )

        cls._loggers['interest'] = LoggerFactory.create_logger(
            'interest', cls._run_dir, None,
            'interest.log', console_handler, use_dual_output=False
        )

        cls._loggers['dividend'] = LoggerFactory.create_logger(
            'dividend', cls._run_dir, None,
            'dividend.log', console_handler, use_dual_output=False
        )

        cls._loggers['verification'] = LoggerFactory.create_logger(
            'verification', cls._run_dir, None,
            'verification.log', console_handler, use_dual_output=False
        )

    @classmethod
    def log_validation_error(
        cls,
        round_number: int,
        agent_id: str,
        agent_type: str,
        error_type: str,
        details: str,
        attempted_action: str,
        debug: bool = False
    ):
        """Log validation error with verification."""
        csv_file_path = cls._run_dir / 'validation_errors.csv'
        CSVLogger.log_validation_error(
            logger=cls._loggers['validation_errors'],
            csv_file_path=csv_file_path,
            round_number=round_number,
            agent_id=agent_id,
            agent_type=agent_type,
            error_type=error_type,
            details=details,
            attempted_action=attempted_action,
            debug=debug
        )

    @classmethod
    def log_margin_call(
        cls,
        round_number: int,
        agent_id: str,
        agent_type: str,
        borrowed_shares: float,
        max_borrowable: float,
        action: str,
        excess_shares: float,
        price: float
    ):
        """Log margin call event."""
        CSVLogger.log_margin_call(
            logger=cls._loggers['margin_calls'],
            round_number=round_number,
            agent_id=agent_id,
            agent_type=agent_type,
            borrowed_shares=borrowed_shares,
            max_borrowable=max_borrowable,
            action=action,
            excess_shares=excess_shares,
            price=price
        )

    @classmethod
    def log_structured_decision(cls, entry):
        """Log structured decision."""
        cls._loggers['structured_decisions'].info(entry.to_csv())

    @classmethod
    def log_decision(cls, message: str):
        """Log decision."""
        cls._loggers['decisions'].info(message)

    @classmethod
    def log_simulation(cls, message: str):
        """Log simulation message."""
        cls._loggers['simulation'].info(message)

    @classmethod
    def log_agent(cls, message: str):
        """Log agent message."""
        cls._loggers['agents'].info(message)

    @classmethod
    def log_order_state(cls, message: str):
        """Log order state message."""
        cls._loggers['order_state'].info(message)

    @classmethod
    def log_info_signal(cls, message: str):
        """Log information signal."""
        cls._loggers['info_signals'].info(message)

    @classmethod
    def log_all_agent_states(
        cls,
        agent_repository: 'AgentRepository',
        round_number: int,
        prefix: str = ""
    ):
        """Log states for all agents."""
        logger = cls.get_logger('agents')
        logger.info(f"\n=== {prefix}Round {round_number} Agent States ===")

        # Get current price from context
        current_price = agent_repository.context.current_price

        for agent_id in agent_repository.get_all_agent_ids():
            agent = agent_repository.get_agent(agent_id)
            state = agent_repository.get_agent_state_snapshot(agent_id, current_price)

            # Format and log messages
            messages = LogFormatter._format_agent_state(state, prefix)
            messages.extend(LogFormatter._format_orders(state.orders_by_state))
            messages.extend(LogFormatter._format_trade_summary(state.trade_summary))

            for msg in messages:
                logger.info(msg.message)

    @classmethod
    def log_trade(cls, trade: 'Trade', prefix: str = ""):
        """Log trade execution."""
        cls.log_order_state(f"\n=== {prefix}Trade Execution ===")
        cls.log_order_state(f"Trade: {trade.quantity} @ ${trade.price:.2f}")

    @classmethod
    def log_trades(cls, trades: List['Trade'], round_number: int):
        """Log trade executions."""
        for trade in trades:
            cls.log_order_state(
                f"Round: {round_number}, "
                f"Trade executed - Price: ${trade.price:.2f}, Quantity: {trade.quantity}, "
                f"Buyer: {trade.buyer_id}, Seller: {trade.seller_id}, "
                f"BuyerOrderID: {trade.buyer_order_id}, SellerOrderID: {trade.seller_order_id}"
            )

    @classmethod
    def log_market_state(cls, order_book, round_number: int, state_name: str):
        """Log market state."""
        cls.log_order_state(f"\n=== {state_name} ===")
        order_book.log_order_book_state()

    @classmethod
    def get_run_dir(cls) -> Path:
        """Get the run directory path."""
        if cls._run_dir is None:
            raise RuntimeError("LoggingService not initialized. Call initialize() first.")
        return cls._run_dir

    @classmethod
    def get_data_dir(cls) -> Path:
        """Get the data directory path."""
        if cls._data_dir is None:
            raise RuntimeError("LoggingService not initialized. Call initialize() first.")
        return cls._data_dir

    @classmethod
    def get_logger(cls, name: str) -> Optional[logging.Logger]:
        """Get logger by name."""
        if not cls._loggers:
            raise RuntimeError("LoggingService not initialized. Call initialize() first.")
        return cls._loggers.get(name)

    @classmethod
    def log_agent_state(
        cls,
        agent_id: str,
        operation: str,
        amount: Optional[float] = None,
        agent_state: dict = None,
        outstanding_orders: dict = None,
        order_history: List = None,
        print_to_terminal: bool = False,
        is_error: bool = False
    ):
        """Log agent state with consistent formatting."""
        logger = cls.get_logger('agents')
        if logger is None:
            raise RuntimeError("Agent logger not initialized")

        # Use the agent_type from the agent_state if provided
        agent_type = agent_state.get('agent_type', None) if agent_state else None

        entry = AgentStateLogEntry(
            agent_id=agent_id,
            operation=operation,
            amount=amount,
            total_cash=agent_state['total_cash'],
            available_cash=agent_state['available_cash'],
            committed_cash=agent_state['committed_cash'],
            dividend_cash=agent_state['dividend_cash'],
            total_shares=agent_state['total_shares'],
            available_shares=agent_state['available_shares'],
            committed_shares=agent_state['committed_shares'],
            outstanding_orders=outstanding_orders,
            order_history=order_history
        )

        # Combine all messages into a single string to prevent splitting
        messages = LogFormatter.format_agent_state(entry)

        # Add agent type to the first message if available
        if agent_type and messages:
            messages[0].message = messages[0].message.replace(
                f"Agent {agent_id}",
                f"Agent {agent_id} ({agent_type})"
            )

        full_message = ''.join(msg.message for msg in messages)

        # Log as a single message
        if is_error or any(msg.level == 'error' for msg in messages):
            logger.error(full_message)
        else:
            logger.info(full_message)

        if print_to_terminal:
            print(full_message)
