from dataclasses import dataclass
from typing import NamedTuple, Union, Dict
from agents.agent_manager.services.agent_data_structures import PositionUpdate

def log_position_update(logger, update: PositionUpdate):
    """Log position changes"""
    logger.info(
        f"Updated {update.agent_id}: "
        f"cash {update.cash_change:+.2f}, "
        f"shares {update.shares_change:+d}"
    )


def update_position_after_trade(position_calculator, agent_repository, trade):
    """Update positions using calculator and repository"""
    # Calculate position changes
    impact = position_calculator.trade_impact_on_positions(trade)
    # Update through repository
    buyer_update = agent_repository.update_agent_position_after_trade(
        trade.buyer_id,
        impact.buyer
    )
    seller_update = agent_repository.update_agent_position_after_trade(
        trade.seller_id,
        impact.seller
    )
    return buyer_update, seller_update

class PositionChange(NamedTuple):
    """Represents a change in an agent's position"""
    cash_change: float
    shares_change: int
    stock_id: str = "DEFAULT_STOCK"  # Default for backwards compatibility
    fee: float = 0.0  # Transaction fee included in cash_change

@dataclass
class TradeImpact:
    """Represents the impact of a trade on both parties"""
    buyer: PositionChange
    seller: PositionChange

class PositionCalculator:
    """Handles position calculations independently"""

    def __init__(self, transaction_cost: Union[float, Dict[str, float]] = 0.0):
        """
        Args:
            transaction_cost: Proportional fee rate charged to each counterparty
                per trade (fee = rate * price * quantity). Either a single float
                or a dict of {stock_id: rate} for multi-stock mode.
        """
        self.transaction_cost = transaction_cost

    def get_fee_rate(self, stock_id: str) -> float:
        """Fee rate for a given stock"""
        if isinstance(self.transaction_cost, dict):
            return self.transaction_cost.get(stock_id, 0.0)
        return self.transaction_cost or 0.0

    def trade_impact_on_positions(self, trade) -> TradeImpact:
        """Calculate position changes from a trade.

        Both counterparties pay fee = rate * trade_value:
        the buyer pays trade_value + fee, the seller receives trade_value - fee.
        """
        trade_value = trade.quantity * trade.price
        assert trade_value == trade.value, "Trade value mismatch"

        fee = self.get_fee_rate(trade.stock_id) * trade_value

        return TradeImpact(
            buyer=PositionChange(
                cash_change=-(trade_value + fee),
                shares_change=trade.quantity,
                stock_id=trade.stock_id,  # Include stock_id for multi-stock support
                fee=fee
            ),
            seller=PositionChange(
                cash_change=trade_value - fee,
                shares_change=0,  # Share reduction already handled during commitment creation (commit_shares)
                stock_id=trade.stock_id,  # Include stock_id for multi-stock support
                fee=fee
            )
        )
