from dataclasses import dataclass
from typing import NamedTuple
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

@dataclass
class TradeImpact:
    """Represents the impact of a trade on both parties"""
    buyer: PositionChange
    seller: PositionChange

class PositionCalculator:
    """Handles position calculations independently"""
    
    @staticmethod
    def trade_impact_on_positions(trade) -> TradeImpact:
        """Calculate position changes from a trade"""
        trade_value = trade.quantity * trade.price
        assert trade_value == trade.value, "Trade value mismatch"

        return TradeImpact(
            buyer=PositionChange(
                cash_change=-trade_value,
                shares_change=trade.quantity,
                stock_id=trade.stock_id  # Include stock_id for multi-stock support
            ),
            seller=PositionChange(
                cash_change=trade_value,
                shares_change=0,  # Share reduction already handled during commitment creation (commit_shares)
                stock_id=trade.stock_id  # Include stock_id for multi-stock support
            )
        )