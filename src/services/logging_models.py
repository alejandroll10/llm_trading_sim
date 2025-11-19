from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from market.orders.order import Order
from market.trade import Trade

if TYPE_CHECKING:
    from agents.agent_manager.agent_repository import AgentStateSnapshot


def sanitize_for_csv(text: str) -> str:
    """Sanitize text for safe CSV storage

    Handles:
    - Newlines and carriage returns (break CSV formatting)
    - Tab characters (misalign columns)
    - Quotes and commas (CSV delimiters)
    - Formula injection (security risk in Excel/Sheets)

    Args:
        text: Raw text to sanitize

    Returns:
        Sanitized text safe for CSV storage
    """
    if not text:
        return ''

    # Remove/replace problematic whitespace characters
    text = text.replace('\n', ' | ').replace('\r', '')  # Use | to preserve logical breaks
    text = text.replace('\t', ' ')  # Replace tabs with spaces

    # Replace CSV delimiter characters
    text = text.replace('"', "'")  # Replace double quotes with single quotes
    text = text.replace(',', ';')  # Replace commas with semicolons

    # Prevent formula injection (security risk if CSV opened in Excel)
    # If text starts with formula characters, prepend with single quote
    if text and text[0] in ('=', '+', '-', '@', '\t', '\r'):
        text = "'" + text

    return text


@dataclass
class ValidationErrorEntry:
    """Structured format for validation error logging"""
    timestamp: str
    round_number: int
    agent_id: str
    agent_type: str
    error_type: str
    details: str
    attempted_action: str

    @staticmethod
    def create_entry(
        round_number: int,
        agent_id: str,
        agent_type: str,
        error_type: str,
        details: str,
        attempted_action: str
    ) -> 'ValidationErrorEntry':
        return ValidationErrorEntry(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            round_number=round_number,
            agent_id=agent_id,
            agent_type=agent_type,
            error_type=error_type,
            details=details.replace(',', ';'),
            attempted_action=attempted_action
        )

    def to_csv(self) -> str:
        """Convert to CSV format"""
        return (
            f"{self.timestamp},{self.round_number},{self.agent_id},"
            f"{self.agent_type},{self.error_type},{self.details},{self.attempted_action}"
        )

@dataclass
class DecisionLogEntry:
    """Structured format for decision logging"""
    timestamp: str
    round_number: int
    agent_id: str
    agent_type_name: str
    agent_type_id: str
    decision: str  # Buy/Sell/Hold
    order_type: str
    quantity: int
    price: float
    reasoning: str
    valuation: float = 0.0  # Agent's estimated fundamental value
    price_target: float = 0.0  # Agent's predicted price for next round
    valuation_reasoning: str = ""  # Separate reasoning for valuation
    price_target_reasoning: str = ""  # Separate reasoning for price target
    notes_to_self: str = ""  # Agent's memory notes for future rounds

    @staticmethod
    def from_decision(
        decision: Dict[str, Any],
        agent_type_name: str,
        agent_type_id: str,
        round_number: int,
        market_price: float
    ) -> List['DecisionLogEntry']:
        """Create log entries from a decision dictionary"""
        entries = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Make sure all fields have defaults to prevent validation errors
        decision_dict = decision.copy()  # Create a copy to avoid modifying the original
        
        # Extract valuation fields, defaulting to 0 if not present
        valuation = decision_dict.get('valuation', 0.0)
        price_target = decision_dict.get('price_target', 0.0)
        valuation_reasoning = sanitize_for_csv(decision_dict.get('valuation_reasoning', ''))
        price_target_reasoning = sanitize_for_csv(decision_dict.get('price_target_reasoning', ''))

        # Ensure reasoning is present
        reasoning = sanitize_for_csv(decision_dict.get('reasoning', ''))

        # Extract memory notes (handle None case when LLM doesn't include field)
        # If notes_to_self is not in decision (feature disabled), use empty string
        notes_to_self = sanitize_for_csv(decision_dict.get('notes_to_self') or '') if 'notes_to_self' in decision_dict else ''
        
        if not decision_dict.get('orders'):
            # Log hold decision
            entries.append(DecisionLogEntry(
                timestamp=timestamp,
                round_number=round_number,
                agent_id=decision_dict.get('agent_id', 'unknown'),
                agent_type_name=agent_type_name,
                agent_type_id=agent_type_id,
                decision='Hold',
                order_type='none',
                quantity=0,
                price=market_price,
                reasoning=reasoning,
                valuation=valuation,
                price_target=price_target,
                valuation_reasoning=valuation_reasoning,
                price_target_reasoning=price_target_reasoning,
                notes_to_self=notes_to_self
            ))
            return entries
            
        # Log each order in the decision
        for order in decision_dict['orders']:
            entries.append(DecisionLogEntry(
                timestamp=timestamp,
                round_number=round_number,
                agent_id=decision_dict.get('agent_id', 'unknown'),
                agent_type_name=agent_type_name,
                agent_type_id=agent_type_id,
                decision=order['decision'],
                order_type=order['order_type'],
                quantity=order['quantity'],
                price=order.get('price_limit', market_price),
                reasoning=reasoning,
                valuation=valuation,
                price_target=price_target,
                valuation_reasoning=valuation_reasoning,
                price_target_reasoning=price_target_reasoning,
                notes_to_self=notes_to_self
            ))
        
        return entries

    def to_csv(self) -> str:
        """Convert to CSV format"""
        return (
            f"{self.timestamp},{self.round_number},{self.agent_id},"
            f"{self.agent_type_name},{self.agent_type_id},{self.decision},"
            f"{self.order_type},{self.quantity},{self.price},\"{self.reasoning}\","
            f"{self.valuation},{self.price_target},\"{self.valuation_reasoning}\",\"{self.price_target_reasoning}\",\"{self.notes_to_self}\""
        )

@dataclass
class LogMessage:
    """Represents a formatted log message"""
    message: str
    level: str = 'info'

@dataclass
class AgentStateLogEntry:
    """Structured format for agent state logging"""
    agent_id: str
    operation: str
    amount: Optional[float]
    total_cash: float
    available_cash: float
    committed_cash: float
    dividend_cash: float
    total_shares: int
    available_shares: int
    committed_shares: int
    outstanding_orders: Optional[Dict[str, List[Order]]] = None
    order_history: Optional[List[Order]] = None

class LogFormatter:
    """Formats log messages for agent state and orders"""
    
    @staticmethod
    def _format_agent_state(agent_state: 'AgentStateSnapshot', prefix: str = "", role: str = "") -> List[LogMessage]:
        """Format agent state for logging"""
        messages = []
        header = f"\n========== {prefix}Agent {agent_state.agent_id}"
        if role:
            header += f" ({role})"
        header += " ==========\n"
        
        messages.append(LogMessage(header))
        messages.append(LogMessage("Current state:"))
        messages.append(LogMessage(f" - Cash: {agent_state.cash:.2f}"))
        messages.append(LogMessage(f" - Available cash: {agent_state.cash - agent_state.committed_cash:.2f}"))
        messages.append(LogMessage(f" - Committed cash: {agent_state.committed_cash:.2f}"))
        messages.append(LogMessage(f" - Dividend cash: {agent_state.dividend_cash:.2f}"))
        messages.append(LogMessage(f" - Total shares: {agent_state.shares}"))
        messages.append(LogMessage(f" - Available shares: {agent_state.shares - agent_state.committed_shares}"))
        messages.append(LogMessage(f" - Committed shares: {agent_state.committed_shares}"))
        messages.append(LogMessage(f" - Wealth: {agent_state.wealth:.2f}"))
        
        return messages

    @staticmethod
    def _format_orders(orders_by_state: Dict[str, Dict[str, List[Order]]]) -> List[LogMessage]:
        """Format orders for logging"""
        messages = []
        messages.append(LogMessage("Outstanding orders:"))
        
        # Format buy orders
        buy_orders = []
        for state in ['pending', 'active', 'partially_filled']:
            buy_orders.extend(orders_by_state[state]['buy'])
        messages.append(LogMessage(f" - Buy orders ({len(buy_orders)}):"))
        for order in buy_orders:
            # Handle case where price might be None
            price_str = f"${order.price:.2f}" if order.price is not None else "None"
            messages.append(LogMessage(f"   * {order.quantity} @ {price_str} (ID: {order.order_id})"))
        
        # Format sell orders
        sell_orders = []
        for state in ['pending', 'active', 'partially_filled']:
            sell_orders.extend(orders_by_state[state]['sell'])
        messages.append(LogMessage(f" - Sell orders ({len(sell_orders)}):"))
        for order in sell_orders:
            # Handle case where price might be None
            price_str = f"${order.price:.2f}" if order.price is not None else "None"
            messages.append(LogMessage(f"   * {order.quantity} @ {price_str} (ID: {order.order_id})"))
        
        return messages

    @staticmethod
    def _format_trade_summary(trade_summary: Dict[str, Any]) -> List[LogMessage]:
        """Format trade summary for logging"""
        messages = []
        if trade_summary:
            messages.append(LogMessage("\nTrade Summary:"))
            messages.append(LogMessage(f" - Buys: {trade_summary['buys']} ({trade_summary['buy_volume']} shares)"))
            messages.append(LogMessage(f" - Sells: {trade_summary['sells']} ({trade_summary['sell_volume']} shares)"))
            if trade_summary['avg_buy_price'] > 0:
                messages.append(LogMessage(f" - Average buy price: ${trade_summary['avg_buy_price']:.2f}"))
            if trade_summary['avg_sell_price'] > 0:
                messages.append(LogMessage(f" - Average sell price: ${trade_summary['avg_sell_price']:.2f}"))
        return messages

    @staticmethod
    def format_agent_state(entry: AgentStateLogEntry) -> List[LogMessage]:
        """Format agent state for logging"""
        # Determine if this is an error message
        is_error = "failed" in entry.operation.lower()
        
        messages = [LogMessage(
            message=f"\n========== Agent {entry.agent_id} {entry.operation} ==========",
            level='error' if is_error else 'info'
        )]
        
        if entry.amount is not None:
            messages.append(LogMessage(
                message=f"\nAmount: {entry.amount}",
                level='error' if is_error else 'info'
            ))
            
        messages.extend([
            LogMessage(
                message=line,
                level='error' if is_error else 'info'
            ) for line in [
                f"\nCurrent state:",
                f"\n - Total cash: {entry.total_cash:.2f}",
                f"\n - Available cash: {entry.available_cash:.2f}",
                f"\n - Committed cash: {entry.committed_cash:.2f}",
                f"\n - Dividend cash: {entry.dividend_cash:.2f}",
                f"\n - Total shares: {entry.total_shares}",
                f"\n - Available shares: {entry.available_shares}",
                f"\n - Committed shares: {entry.committed_shares}"
            ]
        ])
        
        if entry.outstanding_orders:
            messages.extend(LogFormatter._format_outstanding_orders(
                entry.outstanding_orders, 
                is_error=is_error
            ))
            
        if entry.order_history:
            messages.extend(LogFormatter._format_order_history(
                entry.order_history,
                is_error=is_error
            ))
            
        return messages

    @staticmethod
    def _format_outstanding_orders(orders: Dict[str, List[Order]], is_error: bool = False) -> List[LogMessage]:
        """Format outstanding orders"""
        messages = []
        for side, order_list in orders.items():
            messages.extend([
                LogMessage(
                    message=f"\n{side.capitalize()} orders ({len(order_list)}):",
                    level='error' if is_error else 'info'
                )
            ])
            for order in order_list:
                messages.extend([
                    LogMessage(
                        message=f"\n   * {order}",
                        level='error' if is_error else 'info'
                    ),
                    LogMessage(
                        message=f"\n     State: {order.state}",
                        level='error' if is_error else 'info'
                    )
                ])
        return messages

    @staticmethod
    def _format_order_history(orders: List[Order], is_error: bool = False) -> List[LogMessage]:
        """Format order history"""
        messages = [LogMessage(
            message=f"\nOrder history ({len(orders)}):",
            level='error' if is_error else 'info'
        )]
        for order in orders:
            messages.extend([
                LogMessage(
                    message=f"\n * {order}",
                    level='error' if is_error else 'info'
                ),
                LogMessage(
                    message=f"\n   History: {order.print_history()}",
                    level='error' if is_error else 'info'
                )
            ])
        return messages 