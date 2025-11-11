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
        LoggingService.get_logger('agents').info(
            f"Updating share balance for agent {agent_id}, amount: {amount}, stock: {stock_id}")
        agent = self.get_agent(agent_id)

        # Get current positions for this stock
        current_shares = agent.positions.get(stock_id, 0)
        current_borrowed = agent.borrowed_positions.get(stock_id, 0)

        if amount > 0 and current_borrowed > 0:
            # Use purchases to cover existing borrowed shares first
            cover = min(amount, current_borrowed)
            agent.borrowed_positions[stock_id] = current_borrowed - cover
            agent.positions[stock_id] = current_shares + (amount - cover)
            if cover > 0:
                self.borrowing_repository.release_shares(agent_id, cover)
        else:
            agent.positions[stock_id] = current_shares + amount

        LoggingService.get_logger('agents').info(
            f"Updated share balance for agent {agent_id}, stock {stock_id}, new balance: {agent.positions[stock_id]}")
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

        # Calculate per-stock share commitments
        shares_committed_per_stock = {}
        for order in active_orders:
            if order.side == 'sell':
                stock_id = order.stock_id
                shares_committed_per_stock[stock_id] = \
                    shares_committed_per_stock.get(stock_id, 0) + order.current_share_commitment

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
        return AgentCommitmentState(
            agent_id=agent_id,
            available_cash=agent.available_cash,
            available_shares=agent.available_shares,
            total_cash=agent.cash,
            total_shares=agent.shares,
            committed_cash=agent.committed_cash,
            committed_shares=agent.committed_shares
        )

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
        current_price = self.context.current_price
        round_number = self.context.round_number

        # Save original request for tracking
        original_requested = share_amount

        # Get current position for this specific stock
        current_shares = agent.positions.get(stock_id, 0)

        # Determine if we need to borrow shares
        shares_needed = max(0, share_amount - current_shares)
        allocated_shares = 0

        if shares_needed > 0:
            # Try to allocate shares (respects allow_partial_borrows setting)
            allocated_shares = self.borrowing_repository.allocate_shares(agent_id, shares_needed)

            if allocated_shares == 0:
                # No shares available and/or partial fills disabled
                return CommitmentResult(
                    success=False,
                    message=f"Insufficient lendable shares for {stock_id}: requested {shares_needed}, allocated {allocated_shares}",
                    requested_amount=original_requested
                )

            # Calculate the actual fillable amount
            fillable_shares = current_shares + allocated_shares

            # Check if this is a partial fill
            is_partial = fillable_shares < share_amount

            if is_partial:
                # Adjust the commitment to what we can actually fill
                share_amount = fillable_shares
                LoggingService.get_logger('agents').info(
                    f"Partial borrow fill for agent {agent_id} ({stock_id}): "
                    f"requested {original_requested}, "
                    f"fillable {share_amount} (owned: {current_shares}, borrowed: {allocated_shares})"
                )

        try:
            agent.commit_shares(share_amount, round_number=round_number, current_price=current_price, stock_id=stock_id)

            # Determine if this was a partial fill
            partial_fill = shares_needed > 0 and allocated_shares < shares_needed

            return CommitmentResult(
                success=True,
                message="Shares committed successfully" + (" (partial fill)" if partial_fill else ""),
                committed_amount=share_amount,
                partial_fill=partial_fill,
                requested_amount=original_requested
            )
        except ValueError as e:
            # Roll back any allocated shares on failure
            if allocated_shares > 0:
                self.borrowing_repository.release_shares(agent_id, allocated_shares)
            return CommitmentResult(False, str(e), requested_amount=original_requested)
    
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
        try:
            if cash_amount > 0:
                # NEW: Validate cash commitment feasibility before attempting it
                if prices is not None:
                    can_commit, error_msg = agent.can_commit_cash(cash_amount, prices)
                    if not can_commit:
                        return CommitmentResult(False, error_msg)

                agent.commit_cash(cash_amount, prices=prices)
                return CommitmentResult(True, "Cash committed successfully", cash_amount)
            elif share_amount > 0:  # Changed from if to elif to avoid potential double commits
                return self.commit_shares(agent_id, share_amount, stock_id=stock_id)
            else:
                LoggingService.get_logger('agents').error(f"No amount specified for agent {agent_id} with orders: {agent.get_trade_summary()}")
        except ValueError as e:
            return CommitmentResult(False, str(e))
    
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
        try:
            results = []
            if cash_amount > 0:
                agent._release_cash(cash_amount)
                results.append(f"Cash released: {cash_amount}")
            if share_amount > 0:
                # Track per-stock borrowed shares (not global property which only tracks DEFAULT_STOCK)
                borrowed_before = agent.borrowed_positions.get(stock_id, 0)
                agent._release_shares(share_amount, return_borrowed=return_borrowed, stock_id=stock_id)
                results.append(f"Shares released: {share_amount} ({stock_id})")
                borrowed_after = agent.borrowed_positions.get(stock_id, 0)
                returned = borrowed_before - borrowed_after
                if returned > 0:
                    self.borrowing_repository.release_shares(agent_id, returned)

            if results:
                return CommitmentResult(True, "; ".join(results))
            return CommitmentResult(False, "No amount specified")
        except ValueError as e:
            return CommitmentResult(False, str(e))
    
    def get_agent_state_snapshot(self, agent_id: str, prices) -> AgentStateSnapshot:
        """Get complete snapshot of agent state

        Args:
            agent_id: ID of the agent
            prices: Either a single float (single-stock) or Dict[stock_id, price] (multi-stock)
        """
        agent = self.get_agent(agent_id)
        agent.update_wealth(prices)
        
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
            borrowed_cash=agent.borrowed_cash,
            leverage_interest_paid=agent.leverage_interest_paid,
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

        # Update shares through dedicated method, passing stock_id for multi-stock support
        if changes.shares_change != 0:
            self.update_share_balance(agent_id, changes.shares_change, stock_id=changes.stock_id)

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
        previous_total_shares = agent.total_shares

        # Clear positions for ALL stocks (multi-stock support)
        for stock_id in list(agent.positions.keys()):
            agent.positions[stock_id] = 0
            # NOTE: Don't clear committed_positions - agents may still have active orders!
            # Commitments should only be released when orders are cancelled/filled.

            # Release borrowed shares if any
            if stock_id in agent.borrowed_positions and agent.borrowed_positions[stock_id] > 0:
                borrowed = agent.borrowed_positions[stock_id]
                agent.borrowed_positions[stock_id] = 0
                self.borrowing_repository.release_shares(agent_id, borrowed)

        LoggingService.get_logger('agents').info(
            f"Redeemed all shares for agent {agent_id}, "
            f"previous balance: {previous_total_shares}, new balance: {agent.total_shares}"
        )
        return agent