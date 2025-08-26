from typing import Dict, List, Optional, Iterator
from agents.base_agent import BaseAgent
from market.orders.order import Order
import random
from agents.agent_manager.services.position_services import PositionChange
from market.trade import Trade
from market.information.base_information_services import InformationType, InformationSignal, InfoCapability
from agents.agents_api import TradeDecision
from services.logging_service import LoggingService
from agents.agent_manager.services.agent_data_structures import (
    AgentCommitmentState, CommitmentResult,
    AgentStateSnapshot, PositionUpdate, AgentInfoProfile
)
from agents.agent_manager.services.order_services import is_active
from agents.agent_manager.services.borrowing_repository import BorrowingRepository

class AgentRepository:
    """Manages a collection of agents"""
    def __init__(self, agents: List[BaseAgent], logger, context, borrowing_repository: Optional[BorrowingRepository] = None):
        self._agents: Dict[str, BaseAgent] = {
            agent.agent_id: agent for agent in agents
        }
        self._info_profiles: Dict[str, AgentInfoProfile] = {}
        self.context = context
        self.borrowing_repository = borrowing_repository or BorrowingRepository(logger=LoggingService.get_logger('borrowing'))
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        agent = self._agents.get(agent_id)
        if agent is None:
            raise KeyError(f"Agent not found: {agent_id}")
        return agent
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all agents"""
        return list(self._agents.values())
    
    @property
    def agents(self) -> Dict[str, BaseAgent]:
        """Read-only access to agents dictionary"""
        return self._agents.copy()
    
    def __iter__(self) -> Iterator[BaseAgent]:
        """Make repository iterable"""
        return iter(self._agents.values())
    
    def __len__(self) -> int:
        """Get number of agents"""
        return len(self._agents)
    
    def __contains__(self, agent_id: str) -> bool:
        """Check if agent exists"""
        return agent_id in self._agents
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get agents of specific type"""
        return [
            agent for agent in self._agents.values()
            if agent.agent_type.type_id == agent_type
        ]
    
    def update_all_wealth(self, current_price: float):
        """Update wealth for all agents"""
        for agent in self._agents.values():
            agent.update_wealth(current_price)
    
    def verify_all_states(self, debug: bool = False) -> bool:
        """Verify state consistency for all agents"""
        all_valid = True
        for agent in self._agents.values():
            if not agent.verify_state():
                all_valid = False
                LoggingService.log_agent_state(
                    agent_id=agent.agent_id,
                    operation="Agent verification failed",
                    agent_state=agent._get_state_dict(),
                    outstanding_orders=agent.outstanding_orders,
                    order_history=agent.order_history,
                    print_to_terminal=debug,
                    is_error=True
                )
        return all_valid
    
    def update_account_balance(self, agent_id: str, amount: float, account_type: str, 
                             payment_type: str = None, round_number: int = None):
        """Update specific account balance and record payment history if applicable
        
        Args:
            agent_id: ID of the agent
            amount: Amount to update
            account_type: Type of account ('main' or 'dividend')
            payment_type: Type of payment ('interest', 'dividend', etc.)
            round_number: Current round number for history tracking
        """
        LoggingService.get_logger('agents').info(
            f"Updating account balance for agent {agent_id}, "
            f"amount: {amount}, account_type: {account_type}, "
            f"payment_type: {payment_type}"
        )

        agent = self.get_agent(agent_id)
        

        # Update balance as before
        if account_type == "main":
            agent.cash += amount
        elif account_type == "dividend":
            agent.dividend_cash += amount
        else:
            raise ValueError(f"Unknown account type: {account_type}")
        
        # Record payment history if payment_type is provided
        if payment_type and round_number is not None:
            agent.record_payment(account=account_type, amount=amount, payment_type=payment_type, round_number=round_number)

        LoggingService.get_logger('agents').info(
            f"Updated account balance for agent {agent_id}, "
            f"main {agent.cash}, dividend {agent.dividend_cash}, "
            f"committed {agent.committed_cash}, total: {agent.total_cash}"
        )
        return agent
    
    def update_share_balance(self, agent_id: str, amount: int) -> BaseAgent:
        """Update agent's share balance and handle short covering"""
        LoggingService.get_logger('agents').info(
            f"Updating share balance for agent {agent_id}, amount: {amount}")
        agent = self.get_agent(agent_id)

        if amount > 0 and agent.borrowed_shares > 0:
            # Use purchases to cover existing borrowed shares first
            cover = min(amount, agent.borrowed_shares)
            agent.borrowed_shares -= cover
            agent.shares += amount - cover
            if cover > 0:
                self.borrowing_repository.release_shares(agent_id, cover)
        else:
            agent.shares += amount

        LoggingService.get_logger('agents').info(
            f"Updated share balance for agent {agent_id}, new balance: {agent.shares}")
        return agent


    def record_agent_decision(self, agent_id: str, decision: dict):
        """Record decision in agent's history"""
        agent = self.get_agent(agent_id)
        agent.decision_history.append(decision)
    
    def record_agent_order(self, order: Order):
        """Record order in agent's history"""
        agent = self.get_agent(order.agent_id)
        agent.order_history.append(order)

    def sync_agent_orders(self, agent_id: str, orders: List[Order]) -> None:
        """Sync agent's orders with current state"""
        agent = self.get_agent(agent_id)
        
        active_orders = [order for order in orders if is_active(order)]
        
        # Calculate total commitments using active orders
        total_cash_committed = sum(
            order.current_cash_commitment 
            for order in active_orders 
            if order.side == 'buy'
        )
        total_shares_committed = sum(
            order.current_share_commitment 
            for order in active_orders 
            if order.side == 'sell'
        )
        
        # Verify commitments match agent state
        FLOAT_TOLERANCE = 1e-5
        if abs(total_cash_committed - agent.committed_cash) > FLOAT_TOLERANCE:
            error_msg = [
                f"\nCash commitment mismatch for agent {agent_id}:",
                f"Orders total: {total_cash_committed:.2f}, Agent state: {agent.committed_cash:.2f}",
                f"\nAgent State:",
                f"Cash: {agent.cash:.2f}",
                f"Shares: {agent.shares}",
                f"Committed Cash: {agent.committed_cash:.2f}",
                f"Committed Shares: {agent.committed_shares}",
                f"\nActive Orders ({len(active_orders)}):"
            ]
            
            for order in active_orders:
                error_msg.append(f"\n{order}\nHistory:\n{order.print_history()}")
            
            raise ValueError("\n".join(error_msg))
        
        if total_shares_committed != agent.committed_shares:
            error_msg = [
                f"\nShare commitment mismatch for agent {agent_id}:",
                f"Orders total: {total_shares_committed}, Agent state: {agent.committed_shares}",
                f"\nAgent State:",
                f"Cash: {agent.cash:.2f}",
                f"Shares: {agent.shares}",
                f"Committed Cash: {agent.committed_cash:.2f}",
                f"Committed Shares: {agent.committed_shares}",
                f"\nActive Orders ({len(active_orders)}):"
            ]
            
            for order in active_orders:
                error_msg.append(f"\n{order}\nHistory:\n{order.print_history()}")
            
            raise ValueError("\n".join(error_msg))
        
        # Update orders
        agent.sync_orders(orders)
    
    
    def get_all_agent_ids(self) -> List[str]:
        """Get list of all agent IDs"""
        return list(self._agents.keys())
    
    def get_account_balances(self, agent_id: str) -> dict:
        """Get agent's account balances"""
        agent = self.get_agent(agent_id)
        return {
            'main': agent.cash,
            'dividend': agent.dividend_cash,
            'shares': agent.shares,
            'committed_cash': agent.committed_cash,
            'committed_shares': agent.committed_shares,
            'total_cash': agent.total_cash,
            'total_shares': agent.total_shares,
            'total_available_cash': agent.total_available_cash,
        }
    
    def get_agent_decision(self, agent_id: str, market_state: Dict, history: List, round_number: int) -> dict:
        """Get decision from agent with necessary context
        
        Returns:
            dict: A dictionary representation of TradeDecision containing:
                - orders: List of OrderDetails
                - replace_decision: "Cancel", "Replace", or "Add"
                - reasoning: str
        """
        agent = self.get_agent(agent_id)
        decision = agent.make_decision(market_state, history, round_number)
        
        # Ensure decision is converted to dict format for consistency
        if isinstance(decision, TradeDecision):
            return decision.model_dump()
        return decision
    
    def get_shuffled_agent_ids(self) -> List[str]:
        """Get randomized list of agent IDs"""
        agent_ids = list(self._agents.keys())
        random.shuffle(agent_ids)
        return agent_ids
    
    def get_commitment_state(self, agent_id: str) -> AgentCommitmentState:
        """Get agent's commitment-related state"""
        agent = self.get_agent(agent_id)
        return AgentCommitmentState(
            agent_id=agent_id,
            available_cash=agent.available_cash,
            available_shares=agent.available_shares,
            total_cash=agent.cash,
            total_shares=agent.shares,
            committed_cash=agent.committed_cash,
            committed_shares=agent.committed_shares
        )

    def commit_shares(self, agent_id: str, share_amount: int) -> CommitmentResult:
        """Commit shares for an agent, borrowing if necessary."""
        agent = self.get_agent(agent_id)
        current_price = self.context.current_price
        round_number = self.context.round_number

        # Determine if we need to borrow shares
        shares_needed = max(0, share_amount - agent.shares)
        if shares_needed > 0:
            if not self.borrowing_repository.allocate_shares(agent_id, shares_needed):
                return CommitmentResult(False, f"Insufficient lendable shares: requested {shares_needed}")

        try:
            agent.commit_shares(share_amount, round_number=round_number, current_price=current_price)
            return CommitmentResult(True, "Shares committed successfully", share_amount)
        except ValueError as e:
            # Roll back any allocated shares on failure
            if shares_needed > 0:
                self.borrowing_repository.release_shares(agent_id, shares_needed)
            return CommitmentResult(False, str(e))
    
    def commit_resources(self, agent_id: str, cash_amount: float = 0, share_amount: int = 0) -> CommitmentResult:
        """Commit agent resources with validation"""
        agent = self.get_agent(agent_id)
        try:
            if cash_amount > 0:
                agent.commit_cash(cash_amount)
                return CommitmentResult(True, "Cash committed successfully", cash_amount)
            elif share_amount > 0:  # Changed from if to elif to avoid potential double commits
                return self.commit_shares(agent_id, share_amount)
            else:
                LoggingService.get_logger('agents').error(f"No amount specified for agent {agent_id} with orders: {agent.get_trade_summary()}")
        except ValueError as e:
            return CommitmentResult(False, str(e))
    
    def release_resources(self, agent_id: str, cash_amount: float = 0, share_amount: int = 0,
                          return_borrowed: bool = True) -> CommitmentResult:
        """Release agent resources with validation"""
        agent = self.get_agent(agent_id)
        try:
            results = []
            if cash_amount > 0:
                agent._release_cash(cash_amount)
                results.append(f"Cash released: {cash_amount}")
            if share_amount > 0:
                borrowed_before = agent.borrowed_shares
                agent._release_shares(share_amount, return_borrowed=return_borrowed)
                results.append(f"Shares released: {share_amount}")
                returned = borrowed_before - agent.borrowed_shares
                if returned > 0:
                    self.borrowing_repository.release_shares(agent_id, returned)

            if results:
                return CommitmentResult(True, "; ".join(results))
            return CommitmentResult(False, "No amount specified")
        except ValueError as e:
            return CommitmentResult(False, str(e))
    
    def get_agent_state_snapshot(self, agent_id: str, current_price: float) -> AgentStateSnapshot:
        """Get complete snapshot of agent state"""
        agent = self.get_agent(agent_id)
        agent.update_wealth(current_price)
        
        return AgentStateSnapshot(
            agent_id=agent_id,
            agent_type=agent.agent_type.type_id,
            cash=agent.cash,
            dividend_cash=agent.dividend_cash,
            shares=agent.shares,
            committed_cash=agent.committed_cash,
            committed_shares=agent.committed_shares,
            total_shares=agent.total_shares,
            borrowed_shares=agent.borrowed_shares,
            net_shares=agent.total_shares - agent.borrowed_shares,
            wealth=agent.wealth,
            orders_by_state=agent.orders,
            trade_summary=agent.get_trade_summary()
        )
    
    def get_agent_type(self, agent_id: str) -> str:
        """Get agent type identifier"""
        agent = self.get_agent(agent_id)
        return agent.agent_type.type_id
    
    def update_agent_position_after_trade(self, agent_id: str, changes: PositionChange) -> PositionUpdate:
        """Update agent position and return updated state"""
        agent = self.get_agent(agent_id)
        
        # Update cash through dedicated method
        if changes.cash_change != 0:
            self.update_account_balance(agent_id, changes.cash_change, 'main', payment_type='trade', round_number=self.context.round_number)
        
        # Update shares through dedicated method
        if changes.shares_change != 0:
            self.update_share_balance(agent_id, changes.shares_change)
        
        # Replace direct call with LoggingService
        LoggingService.log_agent_state(
            agent_id=agent_id,
            operation="after trade",
            agent_state=agent._get_state_dict(),
            outstanding_orders=agent.outstanding_orders
        )

        return PositionUpdate(
            agent_id=agent_id,
            cash_change=changes.cash_change,
            shares_change=changes.shares_change,
            new_cash=agent.cash,
            new_shares=agent.shares
        )
    
    def record_trade(self, agent_id: str, trade: Trade) -> None:
        """Record trade for the specified agent"""
        agent = self.get_agent(agent_id)
        agent.record_trade(trade)
    
    def distribute_information(self, agent_signals: Dict[str, Dict[InformationType, InformationSignal]]):
        """Distribute information signals to agents"""
        for agent_id, signals in agent_signals.items():
            agent = self.get_agent(agent_id)
            agent.receive_information(signals)

    def redeem_all_shares(self, agent_id: str) -> BaseAgent:
        """Set agent's shares to zero after redemption and clear borrows"""
        agent = self.get_agent(agent_id)
        previous_shares = agent.shares
        agent.shares = 0
        agent.committed_shares = 0

        if agent.borrowed_shares > 0:
            borrowed = agent.borrowed_shares
            agent.borrowed_shares = 0
            self.borrowing_repository.release_shares(agent_id, borrowed)

        LoggingService.get_logger('agents').info(
            f"Redeemed all shares for agent {agent_id}, "
            f"previous balance: {previous_shares}, new balance: {agent.shares}"
        )
        return agent