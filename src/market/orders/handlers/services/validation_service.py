from dataclasses import dataclass
from typing import List, Tuple
from market.orders.order import Order
from services.logging_service import LoggingService

@dataclass
class ValidationResult:
    is_valid: bool
    message: str
    required_commitment: float = 0
    max_executable_quantity: int = 0

class OrderValidationService:
    def __init__(self, commitment_calculator, current_price_provider, agent_manager, logger, order_state_manager):
        self._calculator = commitment_calculator
        self._price_provider = current_price_provider
        self._agent_manager = agent_manager
        self._logger = logger
        self.order_state_manager = order_state_manager
    
    def validate_market_order(self, order: Order) -> ValidationResult:
        """Validate basic market order properties and commitments"""
        if order.order_type != 'market':
            LoggingService.log_validation_error(
                round_number=self._agent_manager.context.round_number,
                agent_id=order.agent_id,
                agent_type=self._agent_manager.agent_repository.get_agent(order.agent_id).__class__.__name__,
                error_type="INVALID_ORDER_TYPE",
                details=f"Expected market order, got {order.order_type}",
                attempted_action=f"{order.order_type.upper()} {order.quantity}"
            )
            return ValidationResult(False, f"Expected market order, got {order.order_type}")
            
        if order.remaining_quantity < 0:
            return ValidationResult(False, "Order quantity cannot be negative")
            
        if order.side not in ['buy', 'sell']:
            return ValidationResult(False, f"Invalid order side: {order.side}")
        
        is_valid, max_qty, message = self._agent_manager.validate_order(order)

        if not is_valid:
            LoggingService.log_validation_error(
                round_number=self._agent_manager.context.round_number,
                agent_id=order.agent_id,
                agent_type=self._agent_manager.agent_repository.get_agent(order.agent_id).__class__.__name__,
                error_type="ORDER_VALIDATION_FAILED",
                details=message,
                attempted_action=f"{order.side.upper()} {order.quantity}"
            )
            return ValidationResult(False, message, max_executable_quantity=0)
        
        required = self._calculator.calculate_required_commitment(
            order, 
            self._price_provider.current_price
        )
        
        if order.side == 'buy' and required > order.current_cash_commitment:
            return ValidationResult(
                False,
                f"Insufficient cash commitment: {order.current_cash_commitment} < {required}",
                required,
                max_executable_quantity=max_qty
            )
        elif order.side == 'sell' and required > order.current_share_commitment:
            return ValidationResult(
                False,
                f"Insufficient share commitment: {order.current_share_commitment} < {required}",
                required,
                max_executable_quantity=max_qty
            )
        
        self.order_state_manager.transition_to_validated(order)
        
        return ValidationResult(True, "Valid", required, max_executable_quantity=max_qty)