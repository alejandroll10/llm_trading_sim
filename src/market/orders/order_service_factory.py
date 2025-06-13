from typing import Tuple
from market.orders.order_state_manager import OrderStateManager
from market.orders.trade_execution_service import TradeExecutionService
from services.shared_service_factory import SharedServiceFactory

class OrderServiceFactory:
    """Factory for creating order-related services with proper dependency management"""
    
    @staticmethod
    def create_services(
        order_repository,
        agent_repository,
        order_book,
        logger
    ) -> Tuple[OrderStateManager, TradeExecutionService]:
        """Create order services with proper dependency injection"""
        
        # Get shared services
        commitment_calculator = SharedServiceFactory.get_commitment_calculator()
        position_calculator = SharedServiceFactory.get_position_calculator()
        
        # Create state manager (independent)
        state_manager = OrderStateManager(
            order_repository=order_repository,
            agent_repository=agent_repository,
            order_book=order_book,
            logger=logger,
            commitment_calculator=commitment_calculator
        )
        
        # Create execution service (depends on state manager)
        trade_service = TradeExecutionService(
            order_repository=order_repository,
            agent_repository=agent_repository,
            order_state_manager=state_manager,  # Inject dependency
            commitment_calculator=commitment_calculator,
            logger=logger,
            position_calculator=position_calculator
        )
        
        return state_manager, trade_service