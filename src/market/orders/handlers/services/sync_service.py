from dataclasses import dataclass
from typing import List, Optional, Dict
from market.orders.order import Order, OrderState

@dataclass
class OrderSyncState:
    active_orders: List[Order]
    committed_cash: float
    committed_shares: int

@dataclass
class SyncResult:
    success: bool
    message: str
    agent_id: str
    orders_synced: int = 0
    error: Optional[Exception] = None

class AgentSyncService:
    """Service to handle synchronization between agents and their orders"""
    
    def __init__(self, agent_repository, order_repository, logger):
        self._agent_repository = agent_repository
        self._order_repository = order_repository
        self._logger = logger
        
    def sync_agent_orders_from_order_repository(self, agent_id: str) -> SyncResult:
        """Synchronize agent's orders with the repository"""
        try:
            # Verify agent exists
            if agent_id not in self._agent_repository:
                return SyncResult(
                    success=False,
                    message=f"Agent {agent_id} not found",
                    agent_id=agent_id
                )
                
            # Get active orders from order repository
            agent_orders = self._order_repository.get_agent_orders(agent_id)
            
            # Validate orders before syncing
            if not self._validate_orders(agent_id, agent_orders):
                return SyncResult(
                    success=False,
                    message="Invalid order state detected",
                    agent_id=agent_id
                )
            
            # Calculate sync state
            sync_state = self._calculate_sync_state(agent_orders)
            
            # Sync through repository
            self._agent_repository.sync_agent_orders(agent_id, sync_state.active_orders)
            
            return SyncResult(
                success=True,
                message="Orders synchronized successfully",
                agent_id=agent_id,
                orders_synced=len(agent_orders)
            )
            
        except Exception as e:
            # Only log detailed information during exceptions
            self._logger.error("\n=== Sync Error ===")
            self._logger.error(f"Failed to sync orders for Agent {agent_id}")
            self._logger.error(f"Error: {str(e)}")
            self._logger.error("\nFull agent orders state:")
            
            # Log all orders and their history during error
            for order in agent_orders:
                self._logger.error(f"\nOrder Details:")
                self._logger.error(f"Order: {order}")
                self._logger.error(f"State: {order.state}")
                self._logger.error(f"Commitments:")
                self._logger.error(f"  - Current Cash: {order.current_cash_commitment:.2f}")
                self._logger.error(f"  - Released Cash: {order.released_cash:.2f}")
                self._logger.error(f"  - Original Cash: {order.original_cash_commitment:.2f}")
                self._logger.error("\nOrder History:")
                self._logger.error(order.print_history())
            
            try:
                agent = self._agent_repository.get_agent(agent_id)
                self._logger.error(f"Agent State:")
                self._logger.error(f"  Cash: {agent.cash:.2f}")
                self._logger.error(f"  Committed Cash: {agent.committed_cash:.2f}")
                self._logger.error(f"  Shares: {agent.shares}")
                self._logger.error(f"  Committed Shares: {agent.committed_shares}")
            except Exception as agent_error:
                self._logger.error(f"Could not get agent state: {str(agent_error)}")
            
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                agent_id=agent_id,
                error=e
            )
    
    def _validate_orders(self, agent_id: str, orders: List[Order]) -> bool:
        """Validate orders before syncing"""
        try:
            # Verify all orders belong to agent
            if not all(order.agent_id == agent_id for order in orders):
                self._logger.error(f"Order agent ID mismatch for agent {agent_id}")
                return False
            
            # Verify no duplicate order IDs
            order_ids = [order.order_id for order in orders]
            if len(order_ids) != len(set(order_ids)):
                self._logger.error(f"Duplicate order IDs found for agent {agent_id}")
                return False
                
            return True
            
        except Exception as e:
            self._logger.error(f"Order validation failed: {str(e)}")
            return False
    
    def _calculate_sync_state(self, orders: List[Order]) -> OrderSyncState:
        """Calculate sync state from orders"""
        total_cash_committed = sum(
            order.current_cash_commitment for order in orders 
            if order.side == 'buy'
        )
        total_shares_committed = sum(
            order.current_share_commitment for order in orders 
            if order.side == 'sell'
        )
        
        return OrderSyncState(
            active_orders=orders,
            committed_cash=total_cash_committed,
            committed_shares=total_shares_committed
        )
    
    def sync_multiple_agents(self, agent_ids: List[str]) -> List[SyncResult]:
        """Synchronize orders for multiple agents"""
        return [self.sync_agent_orders_from_order_repository(agent_id) for agent_id in agent_ids]
    
    def sync_all_agents(self) -> List[SyncResult]:
        """Synchronize orders for all agents"""
        agent_ids = self._agent_repository.get_all_agent_ids()
        return self.sync_multiple_agents(agent_ids)