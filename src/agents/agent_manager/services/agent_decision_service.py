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

        # Check if we should use serial execution (for gpt-oss models on remote APIs with rate limits)
        # Local vLLM endpoints can handle parallel requests fine
        from scenarios.base import DEFAULT_LLM_MODEL, DEFAULT_LLM_BASE_URL
        is_local = DEFAULT_LLM_BASE_URL and 'localhost' in DEFAULT_LLM_BASE_URL
        use_serial = 'gpt-oss' in DEFAULT_LLM_MODEL.lower() and not is_local

        decisions = {}
        if use_serial:
            # Serial execution for gpt-oss (more reliable)
            import time as _time
            for i, agent_id in enumerate(agent_ids):
                # Add small delay between requests to avoid rate limiting
                if i > 0:
                    _time.sleep(0.5)  # 500ms delay between requests
                decisions[agent_id] = self.agent_repository.get_agent_decision(
                    agent_id=agent_id,
                    market_state=market_state,
                    history=history,
                    round_number=round_number
                )
        else:
            # Parallel execution for other models (faster)
            with ThreadPoolExecutor(max_workers=min(len(agent_ids), 2)) as executor:
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

                # Get current price for this stock
                if market_state.get('is_multi_stock'):
                    # Multi-stock: Get price for this specific stock
                    if order.stock_id not in market_state['stocks']:
                        available_stocks = list(market_state['stocks'].keys())
                        error_msg = (
                            f"Agent {agent_id} order specifies stock_id='{order.stock_id}' which doesn't exist. "
                            f"Available stocks: {available_stocks}. "
                            f"In multi-stock scenarios, you must specify the correct stock_id for each order."
                        )
                        self.decisions_logger.error(error_msg)
                        # Skip this order
                        continue
                    current_price = market_state['stocks'][order.stock_id]['price']
                else:
                    # Single-stock: Get the single price
                    current_price = market_state['price']

                valid, _ = self.order_state_manager.handle_new_order(order, current_price)
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
        """Create and register new order with defensive validation"""
        try:
            # Extract stock_id and side early for validation
            if isinstance(order_details, dict):
                stock_id_val = order_details.get('stock_id', 'DEFAULT_STOCK')
                side = order_details['decision'].lower()
            else:
                stock_id_val = order_details.stock_id
                side = order_details.decision.lower()

            # DEFENSIVE VALIDATION: Check stock_id exists in agent positions for sell orders
            if side == 'sell':
                agent = self.agent_repository.get_agent(agent_id)
                if stock_id_val not in agent.positions:
                    available_stocks = list(agent.positions.keys())
                    error_msg = (
                        f"Agent {agent_id} cannot sell stock_id='{stock_id_val}' - "
                        f"not found in agent positions. Available stocks: {available_stocks}. "
                        f"This indicates a stock_id mismatch bug."
                    )
                    self.decisions_logger.error(error_msg)
                    raise ValueError(error_msg)

            # Create order after validation
            if isinstance(order_details, dict):
                order = Order(
                    agent_id=agent_id,
                    stock_id=stock_id_val,  # NEW: Multi-stock support
                    order_type=order_details['order_type'].lower(),
                    side=side,
                    quantity=order_details['quantity'],
                    round_placed=self.context.round_number,
                    price=order_details['price_limit']
                )
            else:
                order = Order(
                    agent_id=agent_id,
                    stock_id=order_details.stock_id,  # NEW: Multi-stock support
                    order_type=order_details.order_type.value,
                    side=side,
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