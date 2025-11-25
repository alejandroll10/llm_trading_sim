import random
from market.orders.handlers.market_handler import MarketOrderHandler
from market.orders.handlers.limit_handler import LimitOrderHandler
from market.orders.order import Order
from typing import List
from market.engine.market_result import MarketResult
from market.engine.services.order_processing_service import OrderProcessingService
from market.engine.services.trade_processing_service import TradeProcessingService
from services.logging_service import LoggingService

class MatchingEngine:
    def __init__(self, order_book, agent_manager, agent_repository, order_repository, context, order_state_manager = None, logger=None, trades_logger=None, trade_execution_service=None, is_multi_stock=False, enable_intra_round_margin_checking=False, stock_id="DEFAULT_STOCK"):
        self.order_book = order_book
        self.agent_manager = agent_manager
        self.order_repository = order_repository
        self.order_state_manager = order_state_manager
        self.agent_repository = agent_repository
        self.trade_execution_service = trade_execution_service
        self.context = context
        self.is_multi_stock = is_multi_stock  # Flag to indicate multi-stock mode
        self.enable_intra_round_margin_checking = enable_intra_round_margin_checking  # Flag to enable margin checking during matching
        self.stock_id = stock_id  # Stock identifier for this engine (used in multi-stock margin checking)
        
        # Initialize services
        self.order_processing_service = OrderProcessingService(order_book)
        self.trade_processing_service = TradeProcessingService(
            agent_manager, 
            order_state_manager, 
            order_book, 
            order_repository, 
            agent_repository, 
            context
        )
        
        # Initialize handlers with services
        self.market_order_handler = MarketOrderHandler(
            order_book=order_book,
            agent_manager=agent_manager,
            trade_execution_service=self.trade_execution_service,
            logger=logger,
            context=self.context,
            order_repository=order_repository,
            order_state_manager=order_state_manager
        )
        
        self.limit_order_handler = LimitOrderHandler(
            order_book=order_book,
            agent_manager=agent_manager,
            trade_execution_service=self.trade_execution_service,
            logger=logger,
            order_repository=order_repository,
            order_state_manager=order_state_manager,
            context=self.context
        )

    def match_orders(self, new_orders: List[Order], current_price: float, round_number: int) -> MarketResult:
        """Match orders for a trading round with prioritized processing"""
        # Log pre-match state
        LoggingService.log_market_state(self.order_book, round_number, "Pre-Match State")

        # Separate liquidation orders initiated by broker
        liquidation_orders = [o for o in new_orders if getattr(o, 'liquidation', False)]
        regular_orders = [o for o in new_orders if not getattr(o, 'liquidation', False)]

        trades = []

        # Process liquidation orders with highest priority
        if liquidation_orders:
            liq_trades, _ = self.market_order_handler.process_orders(
                liquidation_orders, current_price, round_number
            )
            trades.extend(
                self.trade_processing_service.process_market_order_results(
                    liq_trades, [], []
                )
            )
            LoggingService.log_market_state(
                self.order_book, round_number, "Post-Liquidation Order State"
            )

        # Split remaining orders by type
        market_orders, limit_orders = self.order_processing_service.split_orders_by_type(regular_orders)
        
        # Handle limit orders
        crossing_orders = self.limit_order_handler.add_non_crossing_orders(limit_orders)
        LoggingService.log_market_state(self.order_book, round_number, "Post-Limit Order State")
        
        # Process market orders
        if market_orders:
            market_trades, new_aggressive_limits = self.market_order_handler.process_orders(
                market_orders, current_price, round_number
            )
            trades.extend(self.trade_processing_service.process_market_order_results(
                market_trades, new_aggressive_limits, limit_orders
            ))
            LoggingService.log_market_state(self.order_book, round_number, "Post-Market Order State")
        
        # Process crossing limit orders
        if crossing_orders:
            limit_trades = self.limit_order_handler.process_orders(crossing_orders)
            trades.extend(self.trade_processing_service.process_limit_order_results(limit_trades))
        
        # Log trades and calculate new price
        LoggingService.log_trades(trades, round_number)
        new_price = self.trade_processing_service.calculate_new_price(trades, current_price)

        # Check for margin violations and create forced orders (if enabled)
        margin_orders = self._check_and_create_margin_call_orders(new_price, round_number)

        if margin_orders:
            LoggingService.get_logger('market').warning(
                f"[MARGIN_CALL] Processing {len(margin_orders)} forced margin call orders"
            )

            # Process margin call orders using the same flow as regular agent orders:
            # 1. Register in repository
            # 2. Validate and commit resources via handle_new_order()
            # 3. Pass to market handler for matching
            validated_margin_orders = []
            for order in margin_orders:
                self.order_repository.create_order(order)

                # Use the same flow as regular orders - validate and commit resources
                success, message = self.order_state_manager.handle_new_order(order, new_price)

                if success:
                    validated_margin_orders.append(order)
                    LoggingService.get_logger('market').info(
                        f"[MARGIN_CALL] Order {order.order_id} validated and committed successfully"
                    )
                else:
                    LoggingService.get_logger('market').error(
                        f"[MARGIN_CALL] Order {order.order_id} failed: {message}"
                    )

            if not validated_margin_orders:
                LoggingService.get_logger('market').warning(
                    "[MARGIN_CALL] No margin call orders could be validated"
                )
            else:
                # Now process the committed orders through the market handler
                # The handler will validate commitments (should pass) and match
                margin_trades, aggressive_limits = self.market_order_handler.process_orders(
                    validated_margin_orders, new_price, round_number
                )

                if margin_trades:
                    # Add to trades list
                    trades.extend(
                        self.trade_processing_service.process_market_order_results(
                            margin_trades, aggressive_limits, []
                        )
                    )

                    # Return shares to lending pool for filled margin call trades
                    self._process_margin_call_share_returns(margin_trades)

                    # Recalculate price after margin call trades
                    new_price = self.trade_processing_service.calculate_new_price(trades, new_price)

                    LoggingService.get_logger('market').info(
                        f"[MARGIN_CALL] Processed {len(margin_trades)} margin call trades, "
                        f"new price: ${new_price:.2f}"
                    )
                else:
                    LoggingService.get_logger('market').warning(
                        f"[MARGIN_CALL] No margin call trades executed. "
                        f"Converted {len(aggressive_limits)} orders to aggressive limits"
                    )

                    # Still process aggressive limits even if no trades
                    if aggressive_limits:
                        trades.extend(
                            self.trade_processing_service.process_market_order_results(
                                [], aggressive_limits, []
                            )
                        )

        # Update agent wealth and log final state
        # In multi-stock mode, wealth is updated centrally with all stock prices
        if not self.is_multi_stock:
            self.agent_repository.update_all_wealth(new_price)
        LoggingService.log_market_state(self.order_book, round_number, "End of Round State")
        
        return MarketResult(
            trades=trades,
            price=new_price,
            volume=sum(t.quantity for t in trades)
        )

    def _check_and_create_margin_call_orders(self, current_price: float, round_number: int) -> List[Order]:
        """Check all agents for margin violations and create forced orders.

        This method is only active when enable_intra_round_margin_checking=True.
        It checks agents' margin requirements at the current price and creates
        forced market buy orders to cover any violations.

        Args:
            current_price: Price after regular trading (for single stock) or dict of prices (multi-stock)
            round_number: Current round number

        Returns:
            List of forced market orders to resolve margin violations (empty if feature disabled)
        """
        # Feature flag check - return early if disabled
        if not self.enable_intra_round_margin_checking:
            return []

        # Log that we're checking margins
        LoggingService.get_logger('market').info(
            f"[MARGIN_CHECK] Checking margins at price ${current_price:.2f}"
        )

        margin_orders = []

        # Single-stock case: simple margin check
        if not self.is_multi_stock:
            # Check each agent for margin violations
            for agent_id in self.agent_repository.get_all_agent_ids():
                agent = self.agent_repository.get_agent(agent_id)

                # Skip agents without short positions
                if agent.borrowed_shares <= 0:
                    continue

                # Check margin requirement at current price
                max_borrowable = agent.margin_service.get_max_borrowable_shares(current_price)

                if agent.borrowed_shares > max_borrowable:
                    # Margin violation detected!
                    excess = agent.borrowed_shares - max_borrowable

                    LoggingService.get_logger('market').warning(
                        f"[MARGIN_VIOLATION] Agent {agent_id}: "
                        f"borrowed {agent.borrowed_shares:.2f}, "
                        f"max allowed {max_borrowable:.2f}, "
                        f"excess {excess:.2f} shares"
                    )

                    # Create forced buy order for excess shares
                    order = self._create_margin_call_order(
                        agent=agent,
                        quantity=excess,
                        price=current_price,
                        round_number=round_number,
                        stock_id="DEFAULT_STOCK"
                    )
                    margin_orders.append(order)

        else:
            # Multi-stock case: check margin for THIS stock only
            # Each matching engine handles its own stock independently
            for agent_id in self.agent_repository.get_all_agent_ids():
                agent = self.agent_repository.get_agent(agent_id)

                # Get borrowed shares for THIS SPECIFIC stock
                borrowed_for_stock = agent.borrowed_positions.get(self.stock_id, 0)

                # Skip agents without short positions for this stock
                if borrowed_for_stock <= 0:
                    continue

                # Check margin requirement at current price for this stock
                max_borrowable = agent.margin_service.get_max_borrowable_shares(current_price)

                if borrowed_for_stock > max_borrowable:
                    # Margin violation for this stock!
                    excess = borrowed_for_stock - max_borrowable

                    LoggingService.get_logger('market').warning(
                        f"[MARGIN_VIOLATION] Agent {agent_id} on {self.stock_id}: "
                        f"borrowed {borrowed_for_stock:.2f}, "
                        f"max allowed {max_borrowable:.2f}, "
                        f"excess {excess:.2f} shares"
                    )

                    # Create forced buy order for this specific stock
                    order = self._create_margin_call_order(
                        agent=agent,
                        quantity=excess,
                        price=current_price,
                        round_number=round_number,
                        stock_id=self.stock_id
                    )
                    margin_orders.append(order)

        return margin_orders

    def _create_margin_call_order(self, agent, quantity: float, price: float,
                                   round_number: int, stock_id: str) -> Order:
        """Create a forced market buy order for margin call.

        Args:
            agent: Agent that violated margin
            quantity: Number of shares to buy (excess borrowed)
            price: Current market price
            round_number: Current round number
            stock_id: Stock identifier

        Returns:
            Order instance marked as margin call
        """
        order = Order(
            agent_id=agent.agent_id,
            order_type='market',
            side='buy',
            quantity=quantity,
            round_placed=round_number,
            stock_id=stock_id,
            is_margin_call=True  # Mark as forced order
        )

        LoggingService.get_logger('market').warning(
            f"[MARGIN_CALL] Created forced buy order: "
            f"Agent {agent.agent_id}, {quantity:.2f} shares @ ${price:.2f}, "
            f"total cost ~${quantity * price:.2f}"
        )

        return order

    def _process_margin_call_share_returns(self, margin_trades: List) -> None:
        """Process margin call trades and automatically return shares to lending pool.

        When margin call orders fill (agent buys shares to cover short), those shares
        should immediately be used to return borrowed shares to the lending pool.

        Args:
            margin_trades: List of filled margin call trades
        """
        if not margin_trades:
            return

        logger = LoggingService.get_logger('market')

        for trade in margin_trades:
            # Only process buyer side (agent covering their short)
            if hasattr(trade, 'buyer_id'):
                agent = self.agent_repository.get_agent(trade.buyer_id)
                stock_id = trade.stock_id
                shares_bought = trade.quantity

                # Check if agent has borrowed shares for this stock
                if stock_id in agent.borrowed_positions and agent.borrowed_positions[stock_id] > 0:
                    # Calculate how many shares to return (min of bought vs borrowed)
                    shares_to_return = min(shares_bought, agent.borrowed_positions[stock_id])

                    if shares_to_return > 0:
                        # Reduce borrowed position
                        agent.borrowed_positions[stock_id] -= shares_to_return

                        # Return shares to lending pool
                        borrowing_repo = self.agent_repository._get_borrowing_repo(stock_id)
                        borrowing_repo.release_shares(agent.agent_id, shares_to_return)

                        logger.warning(
                            f"[MARGIN_CALL_RETURN] Agent {agent.agent_id} returned {shares_to_return:.2f} "
                            f"shares of {stock_id} to lending pool. "
                            f"Remaining borrowed: {agent.borrowed_positions[stock_id]:.2f}"
                        )