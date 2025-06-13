from typing import List, Dict
from market.orders.order import Order
from agents.agents_api import OrderDetails
from concurrent.futures import ThreadPoolExecutor, as_completed


class AgentDecisionService:
    """Service for collecting and processing agent decisions"""
    
    def __init__(
        self,
        agent_repository,
        order_repository,
        order_state_manager,
        agents_logger,
        decisions_logger,
        context
    ):
        self.agent_repository = agent_repository
        self.order_repository = order_repository
        self.order_state_manager = order_state_manager
        self.agents_logger = agents_logger
        self.decisions_logger = decisions_logger
        self.context = context

    def collect_decisions(self, market_state, history, round_number):
        agent_ids = self.agent_repository.get_shuffled_agent_ids()  
        new_orders = []

        # Collect all decisions in parallel
        decisions = {}
        with ThreadPoolExecutor(max_workers=min(len(agent_ids), 10)) as executor:
            future_to_agent = {
                executor.submit(
                    self.agent_repository.get_agent_decision,
                    agent_id=agent_id,
                    market_state=market_state,
                    history=history,
                    round_number=round_number
                ): agent_id 
                for agent_id in agent_ids
            }
            
            for future in as_completed(future_to_agent):
                agent_id = future_to_agent[future]
                decisions[agent_id] = future.result()

        # Process decisions sequentially in random order
        for agent_id in agent_ids:
            decision = decisions[agent_id]
            active_orders = self.order_repository.get_active_orders_from_agent(agent_id)
            
            # Handle replace_decision first
            if decision['replace_decision'] in ['Cancel', 'Replace']:
                # Cancel all existing orders
                self.order_state_manager.handle_agent_all_orders_cancellation(
                    agent_id=agent_id,
                    orders=active_orders,
                    message="Cancelled by agent decision" if decision['replace_decision'] == 'Cancel' else "Replaced by new orders"
                )
            
            # Handle new orders creation (for Replace or Add)
            for order_details in decision['orders']:
                order = self.create_order(order_details, agent_id)
                valid, _ = self.order_state_manager.handle_new_order(order, market_state['price'])
                if valid:
                    new_orders.append(order)
                    self.agent_repository.record_agent_order(order)
            
            self.agent_repository.record_agent_decision(agent_id, decision)
            self._log_decision(decision, agent_id, round_number)
        
        return new_orders

    def _log_decision(self, decision: dict, agent_id: str, round_number: int):
        """Log agent decision"""
        self.decisions_logger.info(f"=== Logging decisions round {round_number} ===")
        self.decisions_logger.info(f"Logging decision for agent {agent_id} in round {round_number}")
        self.decisions_logger.info(
            f"Round {round_number}: Agent {agent_id} decided: {decision}"
        )
        
    def create_order(self, order_details: Dict | OrderDetails, agent_id: str) -> Order:
        """Create and register new order"""
        try:
            # Handle both dict and OrderDetails formats
            if isinstance(order_details, dict):
                order = Order(
                    agent_id=agent_id,
                    order_type=order_details['order_type'].lower(),
                    side=order_details['decision'].lower(),
                    quantity=order_details['quantity'],
                        round_placed=self.context.round_number,
                    price=order_details['price_limit']
                )
            else:
                order = Order(
                    agent_id=agent_id,
                    order_type=order_details.order_type.value,
                    side=order_details.decision.lower(),
                    quantity=order_details.quantity,
                    round_placed=self.context.round_number,
                    price=order_details.price_limit
                )
        except Exception as e:
            self.decisions_logger.error(f"Error creating order for agent {agent_id}: {e}")
            self.decisions_logger.error(f"Order details: {order_details}")
            raise e
        
        self.order_repository.create_order(order)
        return order