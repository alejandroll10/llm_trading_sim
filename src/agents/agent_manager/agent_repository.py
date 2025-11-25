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
from services.agent_state_calculator import calculate_state_snapshot, calculate_commitment_state
from services.agent_resource_manager import (
    commit_shares_with_borrowing, commit_agent_resources, release_agent_resources,
    update_shares_with_covering, redeem_shares_and_return_borrowed
)
from constants import FLOAT_TOLERANCE

class AgentRepository:
    """Manages a collection of agents"""
    def __init__(self, agents: List[BaseAgent], logger, context,
                 borrowing_repository: Optional[BorrowingRepository] = None,
                 borrowing_repositories: Optional[Dict[str, BorrowingRepository]] = None):
        self._agents: Dict[str, BaseAgent] = {
            agent.agent_id: agent for agent in agents
        }
        self._info_profiles: Dict[str, AgentInfoProfile] = {}
        self.context = context

        # Support both single repository (single-stock) and multiple repositories (multi-stock)
        if borrowing_repositories is not None:
            # Multi-stock mode: dict of repositories
            self.borrowing_repositories = borrowing_repositories
            self.borrowing_repository = list(borrowing_repositories.values())[0] if borrowing_repositories else None
            self.is_multi_stock = True
        else:
            # Single-stock mode: single repository
            self.borrowing_repository = borrowing_repository or BorrowingRepository(logger=LoggingService.get_logger('borrowing'))
            self.borrowing_repositories = None
            self.is_multi_stock = False

    def _get_borrowing_repo(self, stock_id: str = "DEFAULT_STOCK") -> BorrowingRepository:
        """Get the appropriate borrowing repository for a given stock.

        Args:
            stock_id: The stock identifier

        Returns:
            BorrowingRepository for the specified stock
        """
        if self.is_multi_stock:
            if stock_id not in self.borrowing_repositories:
                raise KeyError(f"No borrowing repository found for stock: {stock_id}")
            return self.borrowing_repositories[stock_id]
        else:
            return self.borrowing_repository

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
    
    def update_all_wealth(self, prices):
        """Update wealth for all agents

        Args:
            prices: Either a single float (single-stock) or Dict[stock_id, price] (multi-stock)
        """
        for agent in self._agents.values():
            agent.update_wealth(prices)
    
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
                             payment_type: str = None, round_number: int = None, stock_id: str = None):
        """Update specific account balance and record payment history if applicable

        Args:
            agent_id: ID of the agent
            amount: Amount to update
            account_type: Type of account ('main' or 'dividend')
            payment_type: Type of payment ('interest', 'dividend', etc.)
            round_number: Current round number for history tracking
            stock_id: Optional stock identifier for multi-stock scenarios
        """
        LoggingService.get_logger('agents').info(
            f"Updating account balance for agent {agent_id}, "
            f"amount: {amount}, account_type: {account_type}, "
            f"payment_type: {payment_type}, stock_id: {stock_id}"
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
            agent.record_payment(account=account_type, amount=amount, payment_type=payment_type, round_number=round_number, stock_id=stock_id)

        LoggingService.get_logger('agents').info(
            f"Updated account balance for agent {agent_id}, "
            f"main {agent.cash}, dividend {agent.dividend_cash}, "
            f"committed {agent.committed_cash}, total: {agent.total_cash}"
        )
        return agent
    
    def update_share_balance(self, agent_id: str, amount: int, stock_id: str = "DEFAULT_STOCK") -> BaseAgent:
        """Update agent's share balance for a specific stock and handle short covering"""
        agent = self.get_agent(agent_id)
        return update_shares_with_covering(
            agent=agent,
            amount=amount,
            stock_id=stock_id,
            get_borrowing_repo=self._get_borrowing_repo
        )


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

        # Calculate per-stock share commitments
        shares_committed_per_stock = {}
        for order in active_orders:
            if order.side == 'sell':
                stock_id = order.stock_id
                shares_committed_per_stock[stock_id] = \
                    shares_committed_per_stock.get(stock_id, 0) + order.current_share_commitment

        # Verify commitments match agent state
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

        # Check per-stock share commitments (not global committed_shares which may be 0 for DEFAULT_STOCK)
        for stock_id, expected in shares_committed_per_stock.items():
            actual = agent.committed_positions.get(stock_id, 0)
            if abs(expected - actual) > FLOAT_TOLERANCE:
                error_msg = [
                    f"\nShare commitment mismatch for agent {agent_id}, stock {stock_id}:",
                    f"Orders total: {expected}, Agent state: {actual}",
                    f"\nAgent State:",
                    f"Cash: {agent.cash:.2f}",
                    f"Committed positions: {agent.committed_positions}",
                    f"\nActive Orders ({len(active_orders)}):"
                ]

                for order in active_orders:
                    if order.side == 'sell' and order.stock_id == stock_id:
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
        return calculate_commitment_state(agent)

    def commit_shares(self, agent_id: str, share_amount: int, stock_id: str = "DEFAULT_STOCK") -> CommitmentResult:
        """Commit shares for an agent, borrowing if necessary.

        If partial borrows are enabled and insufficient shares are available to borrow,
        the commitment will be reduced to the maximum fillable amount.

        Args:
            agent_id: ID of the agent
            share_amount: Number of shares to commit
            stock_id: Which stock's shares to commit (for multi-stock mode)
        """
        agent = self.get_agent(agent_id)
        return commit_shares_with_borrowing(
            agent=agent,
            share_amount=share_amount,
            stock_id=stock_id,
            get_borrowing_repo=self._get_borrowing_repo,
            current_price=self.context.current_price,
            round_number=self.context.round_number
        )
    
    def commit_resources(self, agent_id: str, cash_amount: float = 0, share_amount: int = 0, stock_id: str = "DEFAULT_STOCK", prices: Optional[Dict[str, float]] = None) -> CommitmentResult:
        """Commit agent resources with validation

        Args:
            agent_id: ID of the agent
            cash_amount: Amount of cash to commit
            share_amount: Number of shares to commit
            stock_id: Which stock's shares to commit (for multi-stock mode)
            prices: Current prices dict for validation (required for leverage validation)
        """
        agent = self.get_agent(agent_id)
        return commit_agent_resources(
            agent=agent,
            cash_amount=cash_amount,
            share_amount=share_amount,
            stock_id=stock_id,
            prices=prices,
            commit_shares_fn=self.commit_shares
        )
    
    def release_resources(self, agent_id: str, cash_amount: float = 0, share_amount: int = 0,
                          return_borrowed: bool = True, stock_id: str = "DEFAULT_STOCK") -> CommitmentResult:
        """Release agent resources with validation

        Args:
            agent_id: ID of the agent
            cash_amount: Amount of cash to release
            share_amount: Number of shares to release
            return_borrowed: Whether to return borrowed shares to the pool
            stock_id: Which stock's shares to release (for multi-stock mode)
        """
        agent = self.get_agent(agent_id)
        return release_agent_resources(
            agent=agent,
            cash_amount=cash_amount,
            share_amount=share_amount,
            return_borrowed=return_borrowed,
            stock_id=stock_id,
            get_borrowing_repo=self._get_borrowing_repo
        )
    
    def get_agent_state_snapshot(self, agent_id: str, prices) -> AgentStateSnapshot:
        """Get complete snapshot of agent state

        Args:
            agent_id: ID of the agent
            prices: Either a single float (single-stock) or Dict[stock_id, price] (multi-stock)
        """
        agent = self.get_agent(agent_id)
        return calculate_state_snapshot(agent, prices)
    
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

        # Update shares through dedicated method, passing stock_id for multi-stock support
        if changes.shares_change != 0:
            self.update_share_balance(agent_id, changes.shares_change, stock_id=changes.stock_id)

        # AUTOMATIC DEBT REPAYMENT: When agent sells shares (receives cash) and has borrowed_cash,
        # use proceeds to repay debt. This prevents agents from going underwater by selling
        # all shares while keeping the cash and ignoring their debt.
        if changes.cash_change > 0 and agent.borrowed_cash > 0:
            # Calculate how much to repay - use all proceeds up to the debt amount
            repayment = min(changes.cash_change, agent.borrowed_cash)
            if repayment > 0:
                agent.borrowed_cash -= repayment
                agent.cash -= repayment
                # Return cash to lending pool
                if hasattr(agent, 'cash_lending_repo') and agent.cash_lending_repo:
                    agent.cash_lending_repo.release_cash(agent.agent_id, repayment)
                # Record repayment for cash conservation tracking
                if self.context:
                    self.context.record_leverage_cash_repaid(
                        amount=repayment,
                        round_number=self.context.round_number
                    )
                print(f"[AUTO_DEBT_REPAY] Agent {agent_id}: Repaid ${repayment:.2f} from sale proceeds. "
                      f"Remaining debt: ${agent.borrowed_cash:.2f}, Cash after repayment: ${agent.cash:.2f}")

        # Replace direct call with LoggingService
        LoggingService.log_agent_state(
            agent_id=agent_id,
            operation="after trade",
            agent_state=agent._get_state_dict(),
            outstanding_orders=agent.outstanding_orders
        )

        # Get the correct shares for the stock that was traded
        new_shares = agent.positions.get(changes.stock_id, 0)

        return PositionUpdate(
            agent_id=agent_id,
            cash_change=changes.cash_change,
            shares_change=changes.shares_change,
            new_cash=agent.cash,
            new_shares=new_shares  # Shares for the specific stock traded
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
        return redeem_shares_and_return_borrowed(
            agent=agent,
            get_borrowing_repo=self._get_borrowing_repo
        )