from dataclasses import dataclass
from typing import Optional
import logging
from market.trade import Trade
from market.orders.order import OrderState
from agents.agent_manager.services.commitment_services import release_for_trade
from agents.agent_manager.services.position_services import PositionCalculator, update_position_after_trade, log_position_update

@dataclass
class TradeValidationResult:
    is_valid: bool
    message: str

class TradeExecutionService:
    def __init__(self, 
                 order_repository,
                 agent_repository,
                 commitment_calculator,
                 order_state_manager,
                 logger: Optional[logging.Logger],
                 position_calculator: PositionCalculator):
        self.order_repository = order_repository
        self.agent_repository = agent_repository
        self.logger = logger or logging.getLogger('trade_execution')
        self.position_calculator = position_calculator
        self.commitment_calculator = commitment_calculator
        self.order_state_manager = order_state_manager

    def execute_trade(self, trade: Trade):
        """Complete trade execution process"""
        try:
            # Get orders
            buy_order = self.order_repository.get_order(trade.buyer_order_id)
            sell_order = self.order_repository.get_order(trade.seller_order_id)
            
            # Validate
            validation = self._validate_trade(trade, buy_order, sell_order)
            if not validation.is_valid:
                self.logger.error(f"Trade validation failed: {validation.message}")
                return False
                
            # Update order quantities
            buy_order.remaining_quantity -= trade.quantity
            buy_order.filled_quantity += trade.quantity
            sell_order.remaining_quantity -= trade.quantity
            sell_order.filled_quantity += trade.quantity
            
            # Handle state transitions

            # Handle buy order state does not release cash or shares"""
            if buy_order.remaining_quantity == 0:
                self.order_repository.transition_state(
                    trade.buyer_order_id, 
                    OrderState.FILLED,
                    filled_qty=trade.quantity,
                    price=trade.price
                )
            elif buy_order.state not in [OrderState.PARTIALLY_FILLED]:
                # First partial fill does not release cash or shares"""
                self.order_repository.transition_state(
                    trade.buyer_order_id, 
                    OrderState.PARTIALLY_FILLED,
                    filled_qty=trade.quantity,
                    price=trade.price
                )
            
            # Handle sell order state does not release cash or shares"""
            if sell_order.remaining_quantity == 0:
                self.order_repository.transition_state(
                    trade.seller_order_id, 
                    OrderState.FILLED,
                    filled_qty=trade.quantity,
                    price=trade.price
                )
            elif sell_order.state not in [OrderState.PARTIALLY_FILLED]:
                # First partial fill does not release cash or shares"""
                self.order_repository.transition_state(
                    trade.seller_order_id, 
                    OrderState.PARTIALLY_FILLED,
                    filled_qty=trade.quantity,
                    price=trade.price
                )
            
            # Release commitments (just unblocks held resources and does not transfer cash or shares)
            release_for_trade(trade=trade, 
                             order_repository=self.order_repository,
                             agent_repository=self.agent_repository, 
                             commitment_calculator=self.commitment_calculator, 
                             logger=self.logger)
            
            # Update positions (this should handle BOTH cash and share transfers)
            self._update_positions_after_trade(trade)
            
            # Sync orders
            self.order_state_manager.sync_agent_orders(trade.buyer_id)
            self.order_state_manager.sync_agent_orders(trade.seller_id)
            
            # Log success
            self.logger.info(
                f"Trade executed: {trade.quantity} @ ${trade.price:.2f} "
                f"(Buyer: {trade.buyer_id}, Seller: {trade.seller_id})"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: Trade: {trade}/n Buy order: {buy_order}/n Sell order: {sell_order}/n Error: {str(e)}")
            raise

    def _validate_trade(self, trade: Trade, buy_order, sell_order) -> TradeValidationResult:
        """Validate trade details"""
        if trade.quantity <= 0:
            return TradeValidationResult(
                False, 
                f"Invalid trade quantity: {trade.quantity}"
            )
        
        if (buy_order.state == OrderState.FILLED or sell_order.state == OrderState.FILLED):
            return TradeValidationResult(
                False,
                f"At least one order is already filled - Buy order: {buy_order.state}, Sell order: {sell_order.state}"
            )

        if (buy_order.remaining_quantity < trade.quantity or 
            sell_order.remaining_quantity < trade.quantity):
            return TradeValidationResult(
                False,
                f"Invalid remaining quantities - Trade: {trade.quantity}, "
                f"Buy remaining: {buy_order.remaining_quantity}, "
                f"Sell remaining: {sell_order.remaining_quantity}"
            )
            
        return TradeValidationResult(True, "Valid trade")

    def _update_positions_after_trade(self, trade: Trade):
        """Update positions using  positions service"""
        buyer_update, seller_update = update_position_after_trade(
            position_calculator=self.position_calculator,
            agent_repository=self.agent_repository,
            trade=trade
        )
        
        # Log through our logger
        log_position_update(self.logger, buyer_update)
        log_position_update(self.logger, seller_update)

    def handle_trade_execution(self, trade: Trade):
        """Handle state transitions and side effects for trade execution"""
        try:
            success = self.execute_trade(trade)
            if not success:
                self.logger.error(f"Trade execution failed for trade: {trade}")
        except Exception as e:
            self.logger.error(f"Trade execution failed with error: {str(e)}")
            self.logger.error(f"Trade: {trade}")
            # self.logger.error(f"Order repository: {self.order_repository.print_all_orders()}")
            raise
