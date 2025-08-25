import logging
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path
from services.logging_models import ValidationErrorEntry, DecisionLogEntry, LogFormatter, LogMessage, AgentStateLogEntry
import shutil

if TYPE_CHECKING:
    from agents.agent_manager.agent_repository import AgentStateSnapshot, AgentRepository
from market.orders.order import Order
from market.trade import Trade
from market.orders.order_book import OrderBook

@dataclass
class LogMessage:
    """Represents a formatted log message"""
    message: str
    level: str = 'info'

class LoggingService:
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
        """Initialize all loggers and directories"""
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
        
        # Create all necessary directories without removing existing ones
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

        # Setup all loggers with both run-specific and latest directories
        cls._setup_validation_logger(console_handler)
        cls._setup_decisions_logger(console_handler)
        cls._setup_simulation_logger(console_handler)
        cls._setup_structured_decisions_logger(console_handler)
        cls._setup_agents_logger(console_handler)
        cls._setup_order_state_logger(console_handler)
        cls._setup_info_signals_logger(console_handler)
        cls._setup_market_logger(console_handler)
        cls._setup_order_book_logger(console_handler)
        cls._setup_interest_logger(console_handler)
        cls._setup_dividend_logger(console_handler)
        cls._setup_borrow_logger(console_handler)
        cls._setup_borrowing_logger(console_handler)
        cls._setup_verification_logger(console_handler)
        cls._setup_margin_call_logger(console_handler)

        # Prevent duplicate messages
        for logger in cls._loggers.values():
            logger.propagate = False

    @classmethod
    def _setup_logger(cls, name: str, console_handler, filename: str, formatter: logging.Formatter = None):
        """Generic logger setup that writes to both run directory and latest directory"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        if formatter is None:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handlers for both directories
        run_handler = logging.FileHandler(cls._run_dir / filename)
        latest_handler = logging.FileHandler(cls._latest_dir / filename)
        
        for handler in [run_handler, latest_handler]:
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        logger.addHandler(console_handler)
        cls._loggers[name] = logger

    @classmethod
    def _setup_validation_logger(cls, console_handler):
        """Setup validation errors logger"""
        # CSV formatter
        csv_formatter = logging.Formatter('%(message)s')
        
        # Write header to both locations
        for path in [cls._run_dir / 'validation_errors.csv', cls._latest_dir / 'validation_errors.csv']:
            if not path.exists() or path.stat().st_size == 0:
                with open(path, 'w') as f:
                    f.write("timestamp,round_number,agent_id,agent_type,error_type,details,attempted_action\n")
        
        cls._setup_logger('validation_errors', console_handler, 'validation_errors.csv', csv_formatter)

    @classmethod
    def _setup_margin_call_logger(cls, console_handler):
        """Setup margin call events logger"""
        csv_formatter = logging.Formatter('%(message)s')
        header = (
            "timestamp,round_number,agent_id,agent_type,borrowed_shares," \
            "max_borrowable,action,excess_shares,price\n"
        )
        for path in [cls._run_dir / 'margin_calls.csv', cls._latest_dir / 'margin_calls.csv']:
            if not path.exists() or path.stat().st_size == 0:
                with open(path, 'w') as f:
                    f.write(header)
        cls._setup_logger('margin_calls', console_handler, 'margin_calls.csv', csv_formatter)

    @classmethod
    def _setup_structured_decisions_logger(cls, console_handler):
        """Setup structured decisions logger"""
        csv_formatter = logging.Formatter('%(message)s')
        
        # Write header to both locations
        header = "timestamp,round,agent_id,agent_type,agent_type_id,decision,order_type,quantity,price,reasoning,valuation,price_target,valuation_reasoning,price_target_reasoning"
        for path in [cls._run_dir / 'structured_decisions.csv', cls._latest_dir / 'structured_decisions.csv']:
            if not path.exists() or path.stat().st_size == 0:
                with open(path, 'w') as f:
                    f.write(f"{header}\n")
        
        cls._setup_logger('structured_decisions', console_handler, 'structured_decisions.csv', csv_formatter)

    @classmethod
    def _setup_decisions_logger(cls, console_handler):
        """Setup decisions logger"""
        logger = logging.getLogger('decisions')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(os.path.join(cls._run_dir, 'decisions.log'), mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        cls._loggers['decisions'] = logger

    @classmethod
    def _setup_simulation_logger(cls, console_handler):
        """Setup simulation logger"""
        logger = logging.getLogger('simulation')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(os.path.join(cls._run_dir, 'simulation.log'), mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        cls._loggers['simulation'] = logger

    @classmethod
    def _setup_agents_logger(cls, console_handler):
        """Setup agents logger"""
        logger = logging.getLogger('agents')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(os.path.join(cls._run_dir, 'agents.log'), mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        cls._loggers['agents'] = logger

    @classmethod
    def _setup_order_state_logger(cls, console_handler):
        """Setup order state logger"""
        logger = logging.getLogger('order_state')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(os.path.join(cls._run_dir, 'order_state.log'), mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        cls._loggers['order_state'] = logger

    @classmethod
    def _setup_info_signals_logger(cls, console_handler):
        """Setup information signals logger"""
        logger = logging.getLogger('info_signals')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(os.path.join(cls._run_dir, 'info_signals.log'), mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        cls._loggers['info_signals'] = logger

    @classmethod
    def _setup_market_logger(cls, console_handler):
        """Setup market logger"""
        market_logger = logging.getLogger('market')
        market_logger.setLevel(logging.INFO)
        
        # File handlers for both directories
        run_handler = logging.FileHandler(cls._run_dir / 'market.log')
        latest_handler = logging.FileHandler(cls._latest_dir / 'market.log')
        
        for handler in [run_handler, latest_handler]:
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            market_logger.addHandler(handler)
            
        market_logger.addHandler(console_handler)
        cls._loggers['market'] = market_logger

    @classmethod
    def _setup_order_book_logger(cls, console_handler):
        """Setup order book logger"""
        order_book_logger = logging.getLogger('order_book')
        order_book_logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(cls._run_dir / 'order_book.log')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        order_book_logger.addHandler(file_handler)
        order_book_logger.addHandler(console_handler)
        cls._loggers['order_book'] = order_book_logger

    @classmethod
    def _setup_interest_logger(cls, console_handler):
        """Setup interest logger"""
        interest_logger = logging.getLogger('interest')
        interest_logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(cls._run_dir / 'interest.log')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        interest_logger.addHandler(file_handler)
        interest_logger.addHandler(console_handler)
        cls._loggers['interest'] = interest_logger

    @classmethod
    def _setup_dividend_logger(cls, console_handler):
        """Setup dividend logger"""
        dividend_logger = logging.getLogger('dividend')
        dividend_logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(cls._run_dir / 'dividend.log')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        dividend_logger.addHandler(file_handler)
        dividend_logger.addHandler(console_handler)
        cls._loggers['dividend'] = dividend_logger

    @classmethod
    def _setup_borrow_logger(cls, console_handler):
        """Setup borrow fee logger"""
        cls._setup_logger('borrow', console_handler, 'borrow.log')

    @classmethod
    def _setup_borrowing_logger(cls, console_handler):
        """Setup share borrowing logger"""
        cls._setup_logger('borrowing', console_handler, 'borrowing.log')

    @classmethod
    def _setup_verification_logger(cls, console_handler):
        """Setup verification logger"""
        verification_logger = logging.getLogger('verification')
        verification_logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(cls._run_dir / 'verification.log')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        verification_logger.addHandler(file_handler)
        verification_logger.addHandler(console_handler)
        cls._loggers['verification'] = verification_logger

    @classmethod
    def log_validation_error(cls, round_number: int, agent_id: str, agent_type: str, 
                           error_type: str, details: str, attempted_action: str, debug = False):
        """Log validation error"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # Create CSV formatted string
        csv_message = f"{timestamp},{round_number},{agent_id},{agent_type},{error_type},{details},{attempted_action}"
        
        # Get the CSV file path
        csv_file_path = os.path.join(cls._run_dir, 'validation_errors.csv')
        
        # Get number of lines before writing using context manager
        with open(csv_file_path, 'r') as f:
            pre_lines = sum(1 for _ in f)
        
        # Log CSV format to file
        cls._loggers['validation_errors'].info(csv_message)
        
        # Force flush the handlers
        for handler in cls._loggers['validation_errors'].handlers:
            handler.flush()
        
        # Get number of lines after writing using context manager
        with open(csv_file_path, 'r') as f:
            post_lines = sum(1 for _ in f)
        
        # Verify the file exists and has more lines
        if not os.path.exists(csv_file_path):
            raise RuntimeError(
                f"Validation errors CSV file not found at {csv_file_path}. "
                "Logger initialization may have failed."
            )
        
        if post_lines <= pre_lines:  # Must have strictly more lines
            raise RuntimeError(
                f"CRITICAL: Validation error not written to CSV file. "
                f"Lines before: {pre_lines}, after: {post_lines}. "
                f"Message was: {csv_message}\n"
                f"Handlers: {cls._loggers['validation_errors'].handlers}\n"
                f"File path: {csv_file_path}"
            )
        if debug:
            # Log human readable format to console
            cls._loggers['validation_errors'].warning(
                f"Validation Error - Agent {agent_id} ({agent_type}) - {error_type}: {details} "
                f"[Attempted: {attempted_action}]"
            )

    @classmethod
    def log_structured_decision(cls, entry):
        """Log structured decision"""
        cls._loggers['structured_decisions'].info(entry.to_csv())

    @classmethod
    def log_margin_call(cls, round_number: int, agent_id: str, agent_type: str,
                        borrowed_shares: float, max_borrowable: float,
                        action: str, excess_shares: float, price: float):
        """Log margin call events for auditability."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        csv_message = (
            f"{timestamp},{round_number},{agent_id},{agent_type},{borrowed_shares},"
            f"{max_borrowable},{action},{excess_shares},{price}"
        )
        cls._loggers['margin_calls'].info(csv_message)

    @classmethod
    def log_decision(cls, message: str):
        """Log decision"""
        cls._loggers['decisions'].info(message)

    @classmethod
    def log_simulation(cls, message: str):
        """Log simulation message"""
        cls._loggers['simulation'].info(message)

    @classmethod
    def log_agent(cls, message: str):
        """Log agent message"""
        cls._loggers['agents'].info(message)

    @classmethod
    def log_order_state(cls, message: str):
        """Log order state message"""
        cls._loggers['order_state'].info(message)

    @classmethod
    def log_info_signal(cls, message: str):
        """Log information signal"""
        cls._loggers['info_signals'].info(message)

    @classmethod
    def log_all_agent_states(cls, agent_repository: 'AgentRepository', round_number: int, prefix: str = ""):
        """Log states for all agents"""
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
        """Log trade execution"""
        cls.log_order_state(f"\n=== {prefix}Trade Execution ===")
        cls.log_order_state(f"Trade: {trade.quantity} @ ${trade.price:.2f}")

    @classmethod
    def log_trades(cls, trades: List['Trade'], round_number: int):
        """Log trade executions"""
        for trade in trades:
            cls.log_order_state(
                f"Round: {round_number}, "
                f"Trade executed - Price: ${trade.price:.2f}, Quantity: {trade.quantity}, "
                f"Buyer: {trade.buyer_id}, Seller: {trade.seller_id}, "
                f"BuyerOrderID: {trade.buyer_order_id}, SellerOrderID: {trade.seller_order_id}"
            )

    @classmethod
    def log_market_state(cls, order_book, round_number: int, state_name: str):
        """Log market state"""
        cls.log_order_state(f"\n=== {state_name} ===")
        order_book.log_order_book_state()

    @classmethod
    def get_run_dir(cls) -> Path:
        """Get the run directory path"""
        if cls._run_dir is None:
            raise RuntimeError("LoggingService not initialized. Call initialize() first.")
        return cls._run_dir

    @classmethod
    def get_data_dir(cls) -> Path:
        """Get the data directory path"""
        if cls._data_dir is None:
            raise RuntimeError("LoggingService not initialized. Call initialize() first.")
        return cls._data_dir

    @classmethod
    def get_logger(cls, name: str) -> Optional[logging.Logger]:
        """Get logger by name"""
        if not cls._loggers:
            raise RuntimeError("LoggingService not initialized. Call initialize() first.")
        return cls._loggers.get(name)

    @classmethod
    def log_agent_state(cls, agent_id: str, operation: str, amount: Optional[float] = None,
                    agent_state: dict = None, outstanding_orders: dict = None,
                    order_history: List = None, print_to_terminal: bool = False,
                    is_error: bool = False):
        """Log agent state with consistent formatting"""
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