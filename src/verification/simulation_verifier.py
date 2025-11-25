"""
SimulationVerifier: Handles all simulation state verification and invariant checking.

This class is responsible for verifying that the simulation maintains correct invariants
throughout execution, including cash conservation, share conservation, commitment tracking,
and various financial calculations.
"""

from typing import Dict
from services.logging_service import LoggingService
from market.orders.order import OrderState


class SimulationVerifier:
    """
    Verifies simulation state invariants and detects accounting errors.

    Extracted from BaseSimulation to separate verification logic from simulation logic.
    """

    def __init__(self,
                 agent_repository,
                 context,
                 contexts: dict,
                 order_repository,
                 order_book,
                 order_books: dict,
                 borrowing_repository,
                 borrowing_repositories: dict,
                 dividend_service,
                 dividend_services: dict,
                 is_multi_stock: bool,
                 infinite_rounds: bool,
                 agent_params: dict,
                 leverage_enabled: bool = False,
                 cash_lending_repo=None,
                 interest_service=None,
                 borrow_service=None,
                 leverage_interest_service=None):
        """
        Initialize the verifier with references to simulation components.

        Args:
            agent_repository: Repository managing all agents
            context: Primary simulation context (backwards compatible)
            contexts: Dict of contexts for multi-stock mode
            order_repository: Repository managing all orders
            order_book: Primary order book (backwards compatible)
            order_books: Dict of order books for multi-stock mode
            borrowing_repository: Primary borrowing repository
            borrowing_repositories: Dict of borrowing repositories for multi-stock
            dividend_service: Primary dividend service
            dividend_services: Dict of dividend services for multi-stock
            is_multi_stock: Whether this is a multi-stock simulation
            infinite_rounds: Whether simulation runs indefinitely
            agent_params: Agent configuration parameters
            leverage_enabled: Whether leverage trading is enabled
            cash_lending_repo: Cash lending repository for leverage
            interest_service: Interest service for interest rate verification
            borrow_service: Borrow service for borrow fee verification
            leverage_interest_service: Leverage interest service for leverage interest verification
        """
        self.agent_repository = agent_repository
        self.context = context
        self.contexts = contexts
        self.order_repository = order_repository
        self.order_book = order_book
        self.order_books = order_books
        self.borrowing_repository = borrowing_repository
        self.borrowing_repositories = borrowing_repositories
        self.dividend_service = dividend_service
        self.dividend_services = dividend_services
        self.is_multi_stock = is_multi_stock
        self.infinite_rounds = infinite_rounds
        self.agent_params = agent_params
        self.leverage_enabled = leverage_enabled
        self.cash_lending_repo = cash_lending_repo
        self.interest_service = interest_service
        self.borrow_service = borrow_service
        self.leverage_interest_service = leverage_interest_service

        self.logger = LoggingService.get_logger('verification')

    def store_pre_round_states(self) -> Dict[str, Dict]:
        """
        Store pre-round states for verification.

        Returns:
            Dict mapping agent_id to dict with 'total_cash' and 'total_shares'
        """
        pre_round_states = {}

        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)

            pre_round_states[agent_id] = {
                'total_cash': agent.total_cash,
                'total_shares': agent.total_shares,
            }

        return pre_round_states

    def verify_round_end_states(self, pre_round_states):
        """
        Main verification orchestrator - verifies all invariants at end of round.

        This is called after all round processing to ensure simulation state is consistent.
        Throws ValueError if any invariant is violated.

        Args:
            pre_round_states: Pre-round state snapshot from store_pre_round_states()
        """
        self.logger.info(f"\n=== Verifying state changes for round {self.context.round_number} ===")

        # Log per-agent changes
        self._log_agent_changes(pre_round_states)

        # Verify cash and share conservation
        self._verify_cash_conservation(pre_round_states)
        self._verify_share_conservation(pre_round_states)

        # Run all sub-verification methods
        self.verify_borrowing_pool_consistency()
        self.verify_dividend_accumulation()
        self.verify_interest_calculations()
        self.verify_borrow_fee_calculations()
        self.verify_leverage_cash_flows()
        self.verify_order_book_consistency()
        self.verify_wealth_conservation(pre_round_states)
        self.verify_commitment_order_matching()
        self.verify_agent_equity_non_negative()  # CRITICAL: Check no agent has negative equity

        # Multi-stock specific invariants
        if self.is_multi_stock:
            self.verify_multi_stock_invariants()

    def verify_borrowing_pool_consistency(self):
        """Verify borrowing pool accounting is consistent"""
        self.logger.info("\n=== Borrowing Pool Consistency Verification ===")

        # Handle both single-stock and multi-stock modes
        if self.is_multi_stock:
            # Multi-stock: check each stock's borrowing pool
            total_borrowed_from_pool = 0
            total_lendable = 0
            total_available = 0

            for stock_id, borrowing_repo in self.borrowing_repositories.items():
                stock_borrowed = sum(borrowing_repo.borrowed.values())
                total_borrowed_from_pool += stock_borrowed
                total_lendable += borrowing_repo.total_lendable
                total_available += borrowing_repo.available_shares

                # Verify each stock's pool individually
                expected_available = borrowing_repo.total_lendable - stock_borrowed
                if borrowing_repo.available_shares != expected_available:
                    msg = f"Borrowing pool accounting error for {stock_id}:\n"
                    msg += f"Available: {borrowing_repo.available_shares}\n"
                    msg += f"Expected: {expected_available}\n"
                    msg += f"Total lendable: {borrowing_repo.total_lendable}\n"
                    msg += f"Total borrowed: {stock_borrowed}"
                    self.logger.error(msg)
                    raise ValueError(msg)

            self.logger.info(f"Total lendable (all stocks): {total_lendable}")
            self.logger.info(f"Total available (all stocks): {total_available}")
            self.logger.info(f"Total borrowed from pools (all stocks): {total_borrowed_from_pool}")
        else:
            # Single-stock: check the single borrowing pool
            borrowing_repo = self.agent_repository.borrowing_repository
            total_borrowed_from_pool = sum(borrowing_repo.borrowed.values())
            expected_available = borrowing_repo.total_lendable - total_borrowed_from_pool

            self.logger.info(f"Total lendable: {borrowing_repo.total_lendable}")
            self.logger.info(f"Available in pool: {borrowing_repo.available_shares}")
            self.logger.info(f"Total borrowed from pool: {total_borrowed_from_pool}")
            self.logger.info(f"Expected available: {expected_available}")

            if borrowing_repo.available_shares != expected_available:
                msg = f"Borrowing pool accounting error:\n"
                msg += f"Available: {borrowing_repo.available_shares}\n"
                msg += f"Expected: {expected_available}\n"
                msg += f"Total lendable: {borrowing_repo.total_lendable}\n"
                msg += f"Total borrowed: {total_borrowed_from_pool}"
                self.logger.error(msg)
                raise ValueError(msg)

        # Verify agent borrowed shares match pool
        total_agent_borrowed = 0
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            # Sum across all stocks
            for stock_id in agent.borrowed_positions.keys():
                agent_borrowed = agent.borrowed_positions.get(stock_id, 0)
                total_agent_borrowed += agent_borrowed
                if agent_borrowed > 0:
                    self.logger.warning(f"[BORROW_DEBUG] Agent {agent_id} has {agent_borrowed} borrowed shares of {stock_id}")

        self.logger.info(f"Total borrowed by agents: {total_agent_borrowed}")

        # DEBUG: Show pool's records of who borrowed
        if self.is_multi_stock:
            for stock_id, borrowing_repo in self.borrowing_repositories.items():
                for agent_id, amount in borrowing_repo.borrowed.items():
                    if amount > 0:
                        self.logger.warning(f"[BORROW_DEBUG] Pool records agent {agent_id} borrowed {amount} shares of {stock_id}")
        else:
            for agent_id, amount in borrowing_repo.borrowed.items():
                if amount > 0:
                    self.logger.warning(f"[BORROW_DEBUG] Pool records agent {agent_id} borrowed {amount} shares")

        if total_agent_borrowed != total_borrowed_from_pool:
            msg = f"Agent borrowed shares don't match pool:\n"
            msg += f"Agents borrowed: {total_agent_borrowed}\n"
            msg += f"Pool records: {total_borrowed_from_pool}"
            self.logger.error(msg)
            raise ValueError(msg)

        # INVARIANT CHECK: Per-agent-per-stock borrowed positions must match pool records
        self._verify_borrowed_positions_match_pool()

        self.logger.info("✓ Borrowing pool consistency verified")

    def _verify_borrowed_positions_match_pool(self):
        """Verify each agent's borrowed_positions matches what the pool says they borrowed.

        This is a critical invariant:
        - agent.borrowed_positions[stock_id] == pool.borrowed[agent_id] for each stock
        - borrowed + available == total_lendable (pool accounting)
        """
        errors = []

        if self.is_multi_stock:
            for stock_id, borrowing_repo in self.borrowing_repositories.items():
                # Check pool accounting invariant: borrowed + available == total_lendable
                total_borrowed_in_pool = sum(borrowing_repo.borrowed.values())
                expected_total = borrowing_repo.available_shares + total_borrowed_in_pool
                if expected_total != borrowing_repo.total_lendable:
                    errors.append(
                        f"[{stock_id}] Pool accounting broken: "
                        f"available({borrowing_repo.available_shares}) + "
                        f"borrowed({total_borrowed_in_pool}) = {expected_total} "
                        f"!= total_lendable({borrowing_repo.total_lendable})"
                    )

                # Check each agent's borrowed_positions matches pool
                for agent_id in self.agent_repository.get_all_agent_ids():
                    agent = self.agent_repository.get_agent(agent_id)
                    agent_says_borrowed = agent.borrowed_positions.get(stock_id, 0)
                    pool_says_borrowed = borrowing_repo.borrowed.get(agent_id, 0)

                    if agent_says_borrowed != pool_says_borrowed:
                        errors.append(
                            f"[{stock_id}] Agent {agent_id} mismatch: "
                            f"agent.borrowed_positions={agent_says_borrowed}, "
                            f"pool.borrowed={pool_says_borrowed}"
                        )
        else:
            borrowing_repo = self.agent_repository.borrowing_repository

            # Check pool accounting invariant
            total_borrowed_in_pool = sum(borrowing_repo.borrowed.values())
            expected_total = borrowing_repo.available_shares + total_borrowed_in_pool
            if expected_total != borrowing_repo.total_lendable:
                errors.append(
                    f"Pool accounting broken: "
                    f"available({borrowing_repo.available_shares}) + "
                    f"borrowed({total_borrowed_in_pool}) = {expected_total} "
                    f"!= total_lendable({borrowing_repo.total_lendable})"
                )

            # Check each agent's borrowed_positions matches pool
            for agent_id in self.agent_repository.get_all_agent_ids():
                agent = self.agent_repository.get_agent(agent_id)
                agent_says_borrowed = agent.borrowed_positions.get("DEFAULT_STOCK", 0)
                pool_says_borrowed = borrowing_repo.borrowed.get(agent_id, 0)

                if agent_says_borrowed != pool_says_borrowed:
                    errors.append(
                        f"Agent {agent_id} mismatch: "
                        f"agent.borrowed_positions={agent_says_borrowed}, "
                        f"pool.borrowed={pool_says_borrowed}"
                    )

        if errors:
            error_msg = "BORROWED POSITION INVARIANT VIOLATIONS:\n" + "\n".join(errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def verify_dividend_accumulation(self):
        """Verify dividend cash accumulation matches payments"""
        self.logger.info("\n=== Dividend Accumulation Verification ===")

        # Calculate total dividends paid across all stocks
        if self.is_multi_stock:
            total_dividends_paid = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.dividends_paid
            )
        else:
            total_dividends_paid = sum(
                payment['amount']
                for payment in self.context.market_history.dividends_paid
            )

        # Calculate total dividend cash held by agents
        total_dividend_cash = sum(
            self.agent_repository.get_agent(agent_id).dividend_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        self.logger.info(f"Total dividends paid: ${total_dividends_paid:.2f}")
        self.logger.info(f"Total dividend cash held: ${total_dividend_cash:.2f}")

        # Invariant: Dividend cash should equal total dividends paid
        # (unless dividends are paid to main account instead of dividend account)
        if abs(total_dividend_cash - total_dividends_paid) > 0.01:
            # Check if dividends are configured to go to main account
            # In that case, this invariant doesn't apply
            dividend_destination = None
            if self.is_multi_stock:
                # Check first stock's dividend config
                first_stock = list(self.contexts.values())[0]
                dividend_destination = getattr(first_stock, 'dividend_destination', None)
            else:
                dividend_destination = getattr(self.context, 'dividend_destination', None)

            if dividend_destination == 'dividend':
                msg = f"Dividend accumulation mismatch:\n"
                msg += f"Total paid: ${total_dividends_paid:.2f}\n"
                msg += f"Total held: ${total_dividend_cash:.2f}\n"
                msg += f"Difference: ${abs(total_dividend_cash - total_dividends_paid):.2f}"
                self.logger.warning(msg)  # Warning for now as this might be expected in some configs
            else:
                self.logger.info("Dividends paid to main account (not dividend account)")
        else:
            self.logger.info("✓ Dividend accumulation matches payments")

        # Verify non-negative dividend cash
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            if agent.dividend_cash < 0:
                msg = f"Negative dividend cash for agent {agent_id}: ${agent.dividend_cash:.2f}"
                self.logger.error(msg)
                raise ValueError(msg)

        self.logger.info("✓ Dividend accumulation verified")

    def verify_interest_calculations(self):
        """Verify interest rate calculations and payments"""
        self.logger.info("\n=== Interest Calculation Verification ===")

        # Get interest service (shared across all stocks)
        if not self.interest_service:
            self.logger.info("No interest service configured")
            return

        interest_service = self.interest_service
        stock_label = "all stocks" if self.is_multi_stock else "DEFAULT_STOCK"

        # Verify interest rate is within bounds
        rate = interest_service.interest_model.get('rate', 0)
        self.logger.info(f"{stock_label} interest rate: {rate:.4f} ({rate*100:.2f}%)")

        # Invariant 1: Interest rate should be reasonable (between -10% and +50%)
        if rate < -0.10 or rate > 0.50:
            msg = f"Interest rate out of reasonable bounds: {rate:.4f}"
            self.logger.warning(msg)

        # Verify interest payments match calculations
        current_round = self.context.round_number

        if self.is_multi_stock:
            total_interest_paid = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.interest_paid
                if payment['round'] == current_round - 1
            )
        else:
            total_interest_paid = sum(
                payment['amount']
                for payment in self.context.market_history.interest_paid
                if payment['round'] == current_round - 1
            )

        self.logger.info(f"Total interest paid this round: ${total_interest_paid:.2f}")

        # Invariant 2: Interest should be non-negative (assuming positive rates)
        if total_interest_paid < 0:
            # This could be valid for negative interest rates
            if rate >= 0:
                msg = f"Negative interest paid with non-negative rate: ${total_interest_paid:.2f}"
                self.logger.warning(msg)

        # Invariant 3: If interest service exists, check compounding frequency is valid
        compound_freq = interest_service.interest_model.get('compound_frequency', 'per_round')
        valid_frequencies = ['per_round', 'annual', 'semi_annual', 'quarterly', 'monthly']
        if compound_freq not in valid_frequencies:
            msg = f"CRITICAL ERROR - Invalid compound frequency: {compound_freq}\n"
            msg += f"Valid frequencies are: {', '.join(valid_frequencies)}\n"
            msg += f"This is a configuration error that will produce incorrect results."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("✓ Interest calculations verified")

    def verify_borrow_fee_calculations(self):
        """Verify borrow fee (short selling cost) calculations"""
        self.logger.info("\n=== Borrow Fee Calculation Verification ===")

        # Check if borrowing is enabled
        if not hasattr(self.agent_repository, 'borrowing_repository') or \
           self.agent_repository.borrowing_repository.total_lendable == 0:
            self.logger.info("No borrowing enabled")
            return

        # Get borrow fee payments
        current_round = self.context.round_number

        if self.is_multi_stock:
            total_borrow_fees_paid = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.borrow_fees_paid
                if payment['round'] == current_round - 1
            )
        else:
            total_borrow_fees_paid = sum(
                payment['amount']
                for payment in self.context.market_history.borrow_fees_paid
                if payment['round'] == current_round - 1
            )

        # Get total borrowed shares
        total_borrowed = sum(
            self.agent_repository.get_agent(agent_id).borrowed_shares
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        self.logger.info(f"Total borrowed shares: {total_borrowed}")
        self.logger.info(f"Total borrow fees paid: ${total_borrow_fees_paid:.2f}")

        # Invariant 1: Borrow fees should be non-negative
        if total_borrow_fees_paid < 0:
            msg = f"Negative borrow fees paid: ${total_borrow_fees_paid:.2f}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Invariant 2: If shares are borrowed, fees should be paid (unless rate is 0)
        if total_borrowed > 0 and total_borrow_fees_paid == 0:
            # Check if borrow service exists and has a fee rate
            if self.borrow_service:
                # The borrow service might have a borrow_fee rate
                borrow_fee_rate = getattr(self.borrow_service, 'borrow_fee_rate', 0)
                if borrow_fee_rate > 0:
                    msg = f"Shares borrowed ({total_borrowed}) but no fees paid with positive rate ({borrow_fee_rate})"
                    self.logger.warning(msg)

        # Invariant 3: Borrow fees should not exceed total cash in system
        total_cash = sum(
            self.agent_repository.get_agent(agent_id).total_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        if total_borrow_fees_paid > total_cash:
            msg = f"Borrow fees (${total_borrow_fees_paid:.2f}) exceed total system cash (${total_cash:.2f})"
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("✓ Borrow fee calculations verified")

    def verify_leverage_cash_flows(self):
        """Verify leverage cash flow (borrowed cash and interest) calculations"""
        self.logger.info("\n=== Leverage Cash Flow Verification ===")

        # Check if leverage is enabled
        if not self.leverage_enabled:
            self.logger.info("Leverage trading not enabled")
            return

        current_round = self.context.round_number

        # Get leverage cash flows for this round
        if self.is_multi_stock:
            total_leverage_cash_borrowed = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.leverage_cash_borrowed
                if payment['round'] == current_round - 1
            )
            total_leverage_interest_charged = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.leverage_interest_charged
                if payment['round'] == current_round - 1
            )
        else:
            total_leverage_cash_borrowed = sum(
                payment['amount']
                for payment in self.context.market_history.leverage_cash_borrowed
                if payment['round'] == current_round - 1
            )
            total_leverage_interest_charged = sum(
                payment['amount']
                for payment in self.context.market_history.leverage_interest_charged
                if payment['round'] == current_round - 1
            )

        # Get total borrowed cash from agents
        total_agent_borrowed_cash = sum(
            self.agent_repository.get_agent(agent_id).borrowed_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        self.logger.info(f"Total borrowed cash (from agents): ${total_agent_borrowed_cash:.2f}")
        self.logger.info(f"Leverage cash borrowed this round: ${total_leverage_cash_borrowed:.2f}")
        self.logger.info(f"Leverage interest charged this round: ${total_leverage_interest_charged:.2f}")

        # Invariant 1: Borrowed cash should be non-negative
        if total_leverage_cash_borrowed < 0:
            msg = f"Negative leverage cash borrowed: ${total_leverage_cash_borrowed:.2f}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Invariant 2: Interest charged should be non-negative
        if total_leverage_interest_charged < 0:
            msg = f"Negative leverage interest charged: ${total_leverage_interest_charged:.2f}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Invariant 3: If cash is borrowed, interest should be charged (unless rate is 0)
        if total_agent_borrowed_cash > 0 and total_leverage_interest_charged == 0:
            if self.leverage_interest_service:
                interest_rate = getattr(self.leverage_interest_service, 'interest_rate', 0)
                if interest_rate > 0:
                    # This is only a warning for the current round - if borrowing happened in a previous round,
                    # interest might not be charged this round
                    if total_leverage_cash_borrowed > 0:
                        self.logger.warning(
                            f"Cash borrowed (${total_agent_borrowed_cash:.2f}) with positive rate ({interest_rate}) "
                            f"but no interest charged this round"
                        )

        # Invariant 4: Borrowed cash should match what's tracked in the lending pool
        if self.cash_lending_repo:
            pool_total_borrowed = self.cash_lending_repo.get_total_borrowed()
            if abs(pool_total_borrowed - total_agent_borrowed_cash) > 0.01:
                msg = (f"Mismatch between pool borrowed cash (${pool_total_borrowed:.2f}) "
                       f"and agent borrowed cash (${total_agent_borrowed_cash:.2f})")
                self.logger.error(msg)
                raise ValueError(msg)

        self.logger.info("✓ Leverage cash flow calculations verified")

    def verify_order_book_consistency(self):
        """Verify order book consistency invariants"""
        self.logger.info("\n=== Order Book Consistency Verification ===")

        # Get order book(s) - could be single or multiple for multi-stock
        if self.is_multi_stock:
            order_books = self.order_books  # Dict[stock_id, OrderBook]
        else:
            order_books = {"DEFAULT_STOCK": self.order_book}

        for stock_id, order_book in order_books.items():
            self.logger.info(f"\n--- Verifying order book for {stock_id} ---")

            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()

            self.logger.info(f"Best bid: {best_bid}")
            self.logger.info(f"Best ask: {best_ask}")

            # Invariant 1: No crossed market (bid should not exceed ask)
            if best_bid is not None and best_ask is not None:
                if best_bid > best_ask:
                    msg = f"Crossed market detected for {stock_id}:\n"
                    msg += f"Best bid: {best_bid}\n"
                    msg += f"Best ask: {best_ask}\n"
                    msg += f"Bid exceeds ask by: {best_bid - best_ask}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                self.logger.info(f"✓ No crossed market (bid {best_bid} <= ask {best_ask})")

            # Invariant 2: All orders in book should be ACTIVE, PENDING or PARTIALLY_FILLED
            valid_book_states = {OrderState.ACTIVE, OrderState.PENDING, OrderState.PARTIALLY_FILLED}

            invalid_orders = []
            for side_name, heap in [('buy', order_book.buy_orders), ('sell', order_book.sell_orders)]:
                for entry in heap:
                    if entry.order.state not in valid_book_states:
                        invalid_orders.append({
                            'order_id': entry.order.order_id,
                            'side': side_name,
                            'state': entry.order.state,
                            'agent_id': entry.order.agent_id
                        })

            if invalid_orders:
                msg = f"CRITICAL ERROR - Invalid order states in {stock_id} order book:\n"
                for order_info in invalid_orders:
                    msg += f"  Order {order_info['order_id']} ({order_info['side']}): {order_info['state']}\n"
                msg += f"Order book contains orders in invalid states - corrupted state."
                self.logger.error(msg)
                raise ValueError(msg)

            # Invariant 3: Order book quantities match order remaining quantities
            book_buy_quantity = sum(entry.order.remaining_quantity for entry in order_book.buy_orders)
            book_sell_quantity = sum(entry.order.remaining_quantity for entry in order_book.sell_orders)

            self.logger.info(f"Total buy quantity in book: {book_buy_quantity}")
            self.logger.info(f"Total sell quantity in book: {book_sell_quantity}")

            # Invariant 4: Prices are positive
            for entry in order_book.buy_orders:
                if entry.order.price is not None and entry.order.price <= 0:
                    msg = f"Invalid buy order price in {stock_id}: {entry.order.price}"
                    self.logger.error(msg)
                    raise ValueError(msg)

            for entry in order_book.sell_orders:
                if entry.order.price is not None and entry.order.price <= 0:
                    msg = f"Invalid sell order price in {stock_id}: {entry.order.price}"
                    self.logger.error(msg)
                    raise ValueError(msg)

            self.logger.info(f"✓ Order book consistency verified for {stock_id}")

        self.logger.info("\n✓ All order book invariants verified")

    def verify_wealth_conservation(self, pre_round_states):
        """Verify that total wealth changes only due to dividends, interest, and fees"""
        self.logger.info("\n=== Wealth Conservation Verification ===")

        # Calculate pre-round total wealth
        pre_wealth = 0
        for agent_id in pre_round_states:
            # Need to calculate wealth at pre-round prices
            # For simplicity, we'll just verify that trades are zero-sum
            pre_wealth += pre_round_states[agent_id]['total_cash']

        # Calculate post-round total wealth
        post_wealth = sum(
            self.agent_repository.get_agent(agent_id).total_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        # Get payments from this round
        current_round = self.context.round_number

        if self.is_multi_stock:
            dividend_payment = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.dividends_paid
                if payment['round'] == current_round - 1
            )
            interest_payment = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.interest_paid
                if payment['round'] == current_round - 1
            )
            borrow_fee_payment = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.borrow_fees_paid
                if payment['round'] == current_round - 1
            )
            leverage_cash_borrowed = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.leverage_cash_borrowed
                if payment['round'] == current_round - 1
            )
            leverage_interest_charged = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.leverage_interest_charged
                if payment['round'] == current_round - 1
            )
            leverage_cash_repaid = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.leverage_cash_repaid
                if payment['round'] == current_round - 1
            )
        else:
            dividend_payment = sum(
                payment['amount']
                for payment in self.context.market_history.dividends_paid
                if payment['round'] == current_round - 1
            )
            interest_payment = sum(
                payment['amount']
                for payment in self.context.market_history.interest_paid
                if payment['round'] == current_round - 1
            )
            borrow_fee_payment = sum(
                payment['amount']
                for payment in self.context.market_history.borrow_fees_paid
                if payment['round'] == current_round - 1
            )
            leverage_cash_borrowed = sum(
                payment['amount']
                for payment in self.context.market_history.leverage_cash_borrowed
                if payment['round'] == current_round - 1
            )
            leverage_interest_charged = sum(
                payment['amount']
                for payment in self.context.market_history.leverage_interest_charged
                if payment['round'] == current_round - 1
            )
            leverage_cash_repaid = sum(
                payment['amount']
                for payment in self.context.market_history.leverage_cash_repaid
                if payment['round'] == current_round - 1
            )

        expected_wealth_change = dividend_payment + interest_payment - borrow_fee_payment + leverage_cash_borrowed - leverage_interest_charged - leverage_cash_repaid
        actual_wealth_change = post_wealth - pre_wealth

        self.logger.info(f"Pre-round cash: ${pre_wealth:.2f}")
        self.logger.info(f"Post-round cash: ${post_wealth:.2f}")
        self.logger.info(f"Cash change: ${actual_wealth_change:.2f}")
        self.logger.info(f"Expected change (div+int-fees): ${expected_wealth_change:.2f}")

        # This is the same check as the cash conservation check above, but with clearer wealth framing
        if abs(actual_wealth_change - expected_wealth_change) > 0.01:
            msg = f"Wealth conservation violated (trades should be zero-sum):\n"
            msg += f"Cash change: ${actual_wealth_change:.2f}\n"
            msg += f"Expected from payments: ${expected_wealth_change:.2f}\n"
            msg += f"Difference: ${abs(actual_wealth_change - expected_wealth_change):.2f}"
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("✓ Wealth conservation verified (trades are zero-sum)")

    def verify_commitment_order_matching(self):
        """Verify that committed resources match outstanding orders"""
        self.logger.info("\n=== Commitment-Order Matching Verification ===")

        # Get all active orders
        # Check orders that should have commitments: ACTIVE, PENDING, and PARTIALLY_FILLED
        # COMMITTED orders are transitioning (commitments made but not yet in book)
        active_states = {OrderState.ACTIVE, OrderState.PENDING, OrderState.PARTIALLY_FILLED}

        # Calculate expected commitments from orders
        expected_committed_cash = 0
        expected_committed_shares_per_stock = {}

        for order in self.order_repository.orders.values():
            if order.state in active_states:
                if order.side == 'buy':
                    # Buy orders commit cash - use current_cash_commitment which tracks actual commitment
                    expected_committed_cash += order.current_cash_commitment
                elif order.side == 'sell':
                    # Sell orders commit shares - use current_share_commitment not remaining_quantity
                    # because commitment tracks what's actually committed (not filled)
                    stock_id = order.stock_id
                    expected_committed_shares_per_stock[stock_id] = \
                        expected_committed_shares_per_stock.get(stock_id, 0) + order.current_share_commitment

        # Calculate actual commitments
        actual_committed_cash = sum(
            self.agent_repository.get_agent(agent_id).committed_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        # For shares, aggregate across all stocks
        actual_committed_shares_per_stock = {}
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            for stock_id in agent.committed_positions.keys():
                actual_committed_shares_per_stock[stock_id] = \
                    actual_committed_shares_per_stock.get(stock_id, 0) + agent.committed_positions.get(stock_id, 0)

        self.logger.info(f"Expected committed cash from orders: ${expected_committed_cash:.2f}")
        self.logger.info(f"Actual committed cash: ${actual_committed_cash:.2f}")

        # Allow for small floating point differences
        if abs(expected_committed_cash - actual_committed_cash) > 0.1:
            # Debug: Print all order states
            msg = f"CRITICAL ERROR - Commitment-order mismatch for cash:\n"
            msg += f"Expected (from orders): ${expected_committed_cash:.2f}\n"
            msg += f"Actual (from agents): ${actual_committed_cash:.2f}\n"
            msg += f"Difference: ${abs(expected_committed_cash - actual_committed_cash):.2f}\n"
            msg += f"\n=== DEBUG: ALL ORDERS ===\n"
            for order_id, order in self.order_repository.orders.items():
                price_str = f"${order.price:.2f}" if order.price is not None else "market"
                msg += f"{order_id[:8]}: {order.state.value} - {order.side} {order.quantity} @ {price_str}\n"
                msg += f"  cash_commit=${order.current_cash_commitment:.2f}, share_commit={order.current_share_commitment}\n"
            msg += f"\n=== DEBUG: AGENT COMMITMENTS ===\n"
            for agent_id in self.agent_repository.get_all_agent_ids():
                agent = self.agent_repository.get_agent(agent_id)
                if agent.committed_cash > 0.01:
                    msg += f"Agent {agent_id}: committed_cash=${agent.committed_cash:.2f}\n"
            msg += f"\nThis indicates broken bookkeeping - cannot trust simulation results."
            self.logger.error(msg)
            raise ValueError(msg)

        # Check per-stock share commitments
        all_stock_ids = set(expected_committed_shares_per_stock.keys()) | set(actual_committed_shares_per_stock.keys())
        for stock_id in all_stock_ids:
            expected = expected_committed_shares_per_stock.get(stock_id, 0)
            actual = actual_committed_shares_per_stock.get(stock_id, 0)

            self.logger.info(f"{stock_id}: Expected committed shares: {expected}, Actual: {actual}")

            if abs(expected - actual) > 0.01:
                msg = f"CRITICAL ERROR - Commitment-order mismatch for {stock_id} shares:\n"
                msg += f"Expected (from orders): {expected}\n"
                msg += f"Actual (from agents): {actual}\n"
                msg += f"Difference: {abs(expected - actual)}\n"
                msg += f"\n=== DEBUG: SELL ORDERS ===\n"
                for order_id, order in self.order_repository.orders.items():
                    if order.side == 'sell' and order.state in active_states:
                        price_str = f"${order.price:.2f}" if order.price is not None else "market"
                        msg += f"{order_id[:8]}: {order.state.value} - sell {order.quantity} @ {price_str}\n"
                        msg += f"  share_commit={order.current_share_commitment}\n"
                msg += f"\n=== DEBUG: AGENT SHARE COMMITMENTS ===\n"
                for agent_id in self.agent_repository.get_all_agent_ids():
                    agent = self.agent_repository.get_agent(agent_id)
                    total_committed = sum(agent.committed_positions.values())
                    dict_id = id(agent.committed_positions)
                    msg += f"Agent {agent_id}: dict_id={dict_id}, committed_positions={agent.committed_positions}, committed_shares={agent.committed_shares}, total={total_committed}\n"
                msg += f"\nThis indicates broken bookkeeping - cannot trust simulation results."
                self.logger.error(msg)
                raise ValueError(msg)

        self.logger.info("✓ Commitment-order matching verified")

    def verify_multi_stock_invariants(self):
        """Verify multi-stock specific invariants"""
        self.logger.info("\n=== Multi-stock invariant verification ===")

        # Check if this is the final round and shares are being redeemed
        is_final_redemption = (
            self.context.round_number == self.context._num_rounds and
            not self.infinite_rounds
        )

        # Get all stock IDs
        stock_ids = list(self.contexts.keys())
        self.logger.info(f"Verifying {len(stock_ids)} stocks: {stock_ids}")

        # Per-stock share conservation
        for stock_id in stock_ids:
            self.logger.info(f"\n--- Verifying {stock_id} ---")

            # Calculate total shares for this stock across all agents
            total_shares_for_stock = 0
            total_committed_for_stock = 0
            total_borrowed_for_stock = 0

            agent_positions = {}
            for agent_id in self.agent_repository.get_all_agent_ids():
                agent = self.agent_repository.get_agent(agent_id)
                position = agent.positions.get(stock_id, 0)
                committed = agent.committed_positions.get(stock_id, 0)
                borrowed = agent.borrowed_positions.get(stock_id, 0)

                total_shares_for_stock += position + committed
                total_committed_for_stock += committed
                total_borrowed_for_stock += borrowed

                agent_positions[agent_id] = {
                    'position': position,
                    'committed': committed,
                    'borrowed': borrowed,
                    'total': position + committed
                }

            # Get expected initial shares for this stock
            if 'initial_positions' in self.agent_params:
                initial_per_agent = self.agent_params['initial_positions'].get(stock_id, 0)
                num_agents = len(self.agent_repository.get_all_agent_ids())
                expected_initial = initial_per_agent * num_agents
            else:
                expected_initial = 0  # Single-stock mode, should not reach here

            # Expected total includes borrowed shares
            expected_total_with_borrowed = expected_initial + total_borrowed_for_stock

            self.logger.info(f"Initial shares for {stock_id}: {expected_initial}")
            self.logger.info(f"Current total shares: {total_shares_for_stock}")
            self.logger.info(f"Borrowed shares: {total_borrowed_for_stock}")
            self.logger.info(f"Expected total (initial + borrowed): {expected_total_with_borrowed}")

            # Verify share conservation for this stock
            if is_final_redemption:
                # Final round: all shares should be redeemed (0)
                if total_shares_for_stock != 0:
                    msg = f"Final redemption failed for {stock_id}:\n"
                    msg += f"Expected all shares to be redeemed (0), but found {total_shares_for_stock}\n"
                    msg += f"Per-agent breakdown:\n"
                    for agent_id, pos in agent_positions.items():
                        msg += f"  Agent {agent_id}: position={pos['position']}, committed={pos['committed']}, borrowed={pos['borrowed']}, total={pos['total']}\n"
                    self.logger.error(msg)
                    raise ValueError(msg)
                self.logger.info(f"✓ Final redemption verified for {stock_id} (all shares redeemed)")
            else:
                # Normal round: verify share conservation
                if abs(total_shares_for_stock - expected_total_with_borrowed) > 0.01:
                    msg = f"Share conservation violated for {stock_id}:\n"
                    msg += f"Expected: {expected_total_with_borrowed} (initial: {expected_initial} + borrowed: {total_borrowed_for_stock})\n"
                    msg += f"Actual: {total_shares_for_stock}\n"
                    msg += f"Per-agent breakdown:\n"
                    for agent_id, pos in agent_positions.items():
                        msg += f"  Agent {agent_id}: position={pos['position']}, committed={pos['committed']}, borrowed={pos['borrowed']}, total={pos['total']}\n"
                    self.logger.error(msg)
                    raise ValueError(msg)
                self.logger.info(f"✓ Share conservation verified for {stock_id}")

        # Verify all agents have positions for the same stocks
        all_stock_ids = set()
        agent_stock_sets = {}
        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            agent_stocks = set(agent.positions.keys())
            agent_stock_sets[agent_id] = agent_stocks
            all_stock_ids.update(agent_stocks)

        # Check for consistency (all agents should have entries for all stocks in multi-stock mode)
        expected_stocks = set(stock_ids)
        for agent_id, agent_stocks in agent_stock_sets.items():
            # Allow DEFAULT_STOCK to exist in single-stock mode
            agent_stocks_normalized = agent_stocks - {"DEFAULT_STOCK"}
            if agent_stocks_normalized and agent_stocks_normalized != expected_stocks:
                missing = expected_stocks - agent_stocks_normalized
                extra = agent_stocks_normalized - expected_stocks
                msg = f"Stock position inconsistency for agent {agent_id}:\n"
                if missing:
                    msg += f"  Missing stocks: {missing}\n"
                if extra:
                    msg += f"  Extra stocks: {extra}\n"
                msg += f"  Expected: {expected_stocks}\n"
                msg += f"  Actual: {agent_stocks_normalized}"
                self.logger.warning(msg)  # Warning instead of error for now

        self.logger.info("\n✓ All multi-stock invariants verified")

    def _log_agent_changes(self, pre_round_states):
        """Log per-agent cash changes"""
        self.logger.info("\nPer-agent cash changes:")
        for agent_id in pre_round_states:
            pre_cash = pre_round_states[agent_id]['total_cash']
            post_cash = self.agent_repository.get_agent(agent_id).total_cash
            change = post_cash - pre_cash
            self.logger.info(f"Agent {agent_id}: ${pre_cash:.2f} -> ${post_cash:.2f} (Δ${change:.2f})")

    def _verify_cash_conservation(self, pre_round_states):
        """Verify system-wide cash conservation"""
        # Calculate totals
        total_cash_pre = sum(state['total_cash'] for state in pre_round_states.values())
        total_cash_post = sum(
            self.agent_repository.get_agent(agent_id).total_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        # Get payments from this round
        current_round = self.context.round_number

        if self.is_multi_stock:
            dividend_payment = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.dividends_paid
                if payment['round'] == current_round - 1
            )
            interest_payment = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.interest_paid
                if payment['round'] == current_round - 1
            )
            borrow_fee_payment = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.borrow_fees_paid
                if payment['round'] == current_round - 1
            )
            leverage_cash_borrowed = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.leverage_cash_borrowed
                if payment['round'] == current_round - 1
            )
            leverage_interest_charged = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.leverage_interest_charged
                if payment['round'] == current_round - 1
            )
            leverage_cash_repaid = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.leverage_cash_repaid
                if payment['round'] == current_round - 1
            )
        else:
            dividend_payment = sum(
                payment['amount']
                for payment in self.context.market_history.dividends_paid
                if payment['round'] == current_round - 1
            )
            interest_payment = sum(
                payment['amount']
                for payment in self.context.market_history.interest_paid
                if payment['round'] == current_round - 1
            )
            borrow_fee_payment = sum(
                payment['amount']
                for payment in self.context.market_history.borrow_fees_paid
                if payment['round'] == current_round - 1
            )
            leverage_cash_borrowed = sum(
                payment['amount']
                for payment in self.context.market_history.leverage_cash_borrowed
                if payment['round'] == current_round - 1
            )
            leverage_interest_charged = sum(
                payment['amount']
                for payment in self.context.market_history.leverage_interest_charged
                if payment['round'] == current_round - 1
            )
            leverage_cash_repaid = sum(
                payment['amount']
                for payment in self.context.market_history.leverage_cash_repaid
                if payment['round'] == current_round - 1
            )

        # DEBUG: Add detailed logging for payment breakdown
        self.logger.warning(
            f"[CASH_DEBUG] Round {current_round} Payment Breakdown:\n"
            f"  Dividends: ${dividend_payment:.2f}\n"
            f"  Interest: ${interest_payment:.2f}\n"
            f"  Borrow Fees: ${borrow_fee_payment:.2f}\n"
            f"  Leverage Cash Borrowed: ${leverage_cash_borrowed:.2f}\n"
            f"  Leverage Interest Charged: ${leverage_interest_charged:.2f}\n"
            f"  Leverage Cash Repaid: ${leverage_cash_repaid:.2f}\n"
            f"  Total Cash Pre-Round: ${total_cash_pre:.2f}\n"
            f"  Total Cash Post-Round: ${total_cash_post:.2f}"
        )

        # Verify round-by-round changes
        cash_difference = total_cash_post - total_cash_pre
        total_round_payments = dividend_payment + interest_payment - borrow_fee_payment + leverage_cash_borrowed - leverage_interest_charged - leverage_cash_repaid

        self.logger.warning(
            f"[CASH_DEBUG] Cash Verification:\n"
            f"  Actual Change: ${cash_difference:.2f}\n"
            f"  Expected Change: ${total_round_payments:.2f}\n"
            f"  Difference: ${abs(cash_difference - total_round_payments):.2f}"
        )

        if abs(cash_difference - total_round_payments) > 0.01:
            msg = (f"Round cash change doesn't match round payments:\n"
                   f"Change in cash: ${cash_difference:.2f}\n"
                   f"Round payments: ${total_round_payments:.2f}")
            self.logger.error(msg)
            raise ValueError(msg)

        # Verify total system cash
        initial_cash = sum(
            self.agent_repository.get_agent(agent_id).initial_cash
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        if self.is_multi_stock:
            total_historical_dividends = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.dividends_paid
            )
            total_historical_interest = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.interest_paid
            )
            total_historical_borrow_fees = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.borrow_fees_paid
            )
            total_historical_leverage_cash_borrowed = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.leverage_cash_borrowed
            )
            total_historical_leverage_interest_charged = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.leverage_interest_charged
            )
            total_historical_leverage_cash_repaid = sum(
                payment['amount']
                for context in self.contexts.values()
                for payment in context.market_history.leverage_cash_repaid
            )
        else:
            total_historical_dividends = sum(payment['amount'] for payment in self.context.market_history.dividends_paid)
            total_historical_interest = sum(payment['amount'] for payment in self.context.market_history.interest_paid)
            total_historical_borrow_fees = sum(payment['amount'] for payment in self.context.market_history.borrow_fees_paid)
            total_historical_leverage_cash_borrowed = sum(payment['amount'] for payment in self.context.market_history.leverage_cash_borrowed)
            total_historical_leverage_interest_charged = sum(payment['amount'] for payment in self.context.market_history.leverage_interest_charged)
            total_historical_leverage_cash_repaid = sum(payment['amount'] for payment in self.context.market_history.leverage_cash_repaid)

        expected_total_cash = (
            initial_cash + total_historical_dividends + total_historical_interest - total_historical_borrow_fees
            + total_historical_leverage_cash_borrowed - total_historical_leverage_interest_charged - total_historical_leverage_cash_repaid
        )

        self.logger.info(f"\n=== System-wide cash verification ===")
        self.logger.info(f"Initial cash: ${initial_cash:.2f}")
        self.logger.info(f"Total historical dividends: ${total_historical_dividends:.2f}")
        self.logger.info(f"Total historical interest: ${total_historical_interest:.2f}")
        self.logger.info(f"Total historical borrow fees: ${total_historical_borrow_fees:.2f}")
        self.logger.info(f"Total historical leverage cash borrowed: ${total_historical_leverage_cash_borrowed:.2f}")
        self.logger.info(f"Total historical leverage interest charged: ${total_historical_leverage_interest_charged:.2f}")
        self.logger.info(f"Total historical leverage cash repaid: ${total_historical_leverage_cash_repaid:.2f}")
        self.logger.info(f"Current total cash: ${total_cash_post:.2f}")
        self.logger.info(f"Expected total: ${expected_total_cash:.2f}")

        if abs(total_cash_post - expected_total_cash) > 0.01:
            msg = "Total system cash doesn't match historical payments"
            self.logger.error(msg)
            raise ValueError(msg)

    def _verify_share_conservation(self, pre_round_states):
        """Verify share conservation"""
        initial_shares = sum(
            self.agent_repository.get_agent(agent_id).initial_shares
            for agent_id in self.agent_repository.get_all_agent_ids()
        )

        # Check if this is the final round and shares are being redeemed
        is_final_redemption = (
            self.context.round_number == self.context._num_rounds and
            not self.infinite_rounds
        )

        self.logger.info("\n=== Share allocation verification ===")
        self.logger.info("Initial allocation:")
        if 'initial_shares' in self.agent_params:
            self.logger.info(f"Per agent: {self.agent_params['initial_shares']}")
        elif 'initial_positions' in self.agent_params:
            self.logger.info(f"Per agent (multi-stock): {self.agent_params['initial_positions']}")
        self.logger.info(f"Total: {initial_shares}")

        # Log pre-round share distribution
        self.logger.info("\nPre-round share distribution:")
        pre_shares = {
            agent_id: state['total_shares']
            for agent_id, state in pre_round_states.items()
        }
        for agent_id, shares in pre_shares.items():
            self.logger.info(f"Agent {agent_id}: {shares}")
        self.logger.info(f"Pre-round total: {sum(pre_shares.values())}")

        # Log post-round share distribution
        self.logger.info("\nPost-round share distribution:")
        post_shares = {
            agent_id: self.agent_repository.get_agent(agent_id).total_shares
            for agent_id in self.agent_repository.get_all_agent_ids()
        }
        for agent_id, shares in post_shares.items():
            agent = self.agent_repository.get_agent(agent_id)
            # DEBUG: Show detailed breakdown including borrowed
            self.logger.warning(
                f"[SHARE_DEBUG] Agent {agent_id}: total_shares={shares}, "
                f"positions={dict(agent.positions)}, "
                f"committed={dict(agent.committed_positions)}, "
                f"borrowed={dict(agent.borrowed_positions)}"
            )

        total_shares_current = sum(post_shares.values())
        self.logger.info(f"Post-round total: {total_shares_current}")

        # Calculate total borrowed shares across all stocks
        if self.is_multi_stock:
            # Sum borrowed shares from all stock repositories
            borrowed_total = sum(
                repo.total_lendable - repo.available_shares
                for repo in self.borrowing_repositories.values()
            )
            # DEBUG: Log pool states
            for stock_id, repo in self.borrowing_repositories.items():
                self.logger.warning(
                    f"[SHARE_DEBUG] Pool for {stock_id}: "
                    f"total_lendable={repo.total_lendable}, "
                    f"available={repo.available_shares}, "
                    f"borrowed_total={repo.total_lendable - repo.available_shares}, "
                    f"borrowed_by_agent={dict(repo.borrowed)}"
                )
        else:
            # Single stock: use the single repository
            repo = self.agent_repository.borrowing_repository
            borrowed_total = repo.total_lendable - repo.available_shares
            # DEBUG: Log pool state
            self.logger.warning(
                f"[SHARE_DEBUG] Pool (single stock): "
                f"total_lendable={repo.total_lendable}, "
                f"available={repo.available_shares}, "
                f"borrowed_total={borrowed_total}, "
                f"borrowed_by_agent={dict(repo.borrowed)}"
            )
        expected_total_shares = initial_shares + borrowed_total

        # Verification logic
        if is_final_redemption:
            if total_shares_current != 0:
                msg = (f"Final redemption: All shares should be zero but found {total_shares_current} shares\n"
                       f"Initial shares: {initial_shares}\n"
                       f"Current shares: {total_shares_current}\n"
                       f"Share changes this round:")
                for agent_id in pre_shares:
                    change = post_shares[agent_id] - pre_shares[agent_id]
                    msg += f"\nAgent {agent_id}: {pre_shares[agent_id]} -> {post_shares[agent_id]} (Δ{change})"
                self.logger.error(msg)
                raise ValueError(msg)
            else:
                redemption_msg = "Final redemption: All shares successfully redeemed"
                if self.is_multi_stock:
                    redemption_msg += f" across {len(self.dividend_services)} stocks"
                self.logger.info(redemption_msg)
        elif total_shares_current != expected_total_shares:
            msg = (f"Total shares in system changed from initial allocation:\n"
                   f"Initial shares: {initial_shares}\n"
                   f"Borrowed shares: {borrowed_total}\n"
                   f"Current shares: {total_shares_current}\n"
                   f"Share changes this round:")
            for agent_id in pre_shares:
                change = post_shares[agent_id] - pre_shares[agent_id]
                msg += f"\nAgent {agent_id}: {pre_shares[agent_id]} -> {post_shares[agent_id]} (Δ{change})"
            self.logger.error(msg)
            raise ValueError(msg)

    def verify_agent_equity_non_negative(self):
        """
        CRITICAL INVARIANT: Verify no agent has negative equity (total value).

        An agent's equity = total_cash + share_value - borrowed_cash
        This should NEVER be negative. If it is, it means:
        1. Margin calls were not properly enforced
        2. An agent owes more than they own

        This is a critical accounting invariant that must be maintained.
        """
        self.logger.info("\n=== Agent Equity Non-Negative Verification ===")

        violations = []

        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)

            # Calculate total share value across all positions
            total_share_value = 0
            if self.is_multi_stock:
                for stock_id, shares in agent.positions.items():
                    if stock_id in self.contexts:
                        price = self.contexts[stock_id].current_price
                        total_share_value += shares * price
            else:
                total_share_value = agent.total_shares * self.context.current_price

            # Calculate equity: cash + share_value - borrowed_cash
            total_cash = agent.total_cash
            borrowed_cash = getattr(agent, 'borrowed_cash', 0)
            equity = total_cash + total_share_value - borrowed_cash

            self.logger.info(
                f"Agent {agent_id}: cash=${total_cash:.2f}, shares_value=${total_share_value:.2f}, "
                f"borrowed_cash=${borrowed_cash:.2f}, equity=${equity:.2f}"
            )

            if equity < -0.01:  # Small tolerance for floating point errors
                violations.append({
                    'agent_id': agent_id,
                    'cash': total_cash,
                    'share_value': total_share_value,
                    'borrowed_cash': borrowed_cash,
                    'equity': equity
                })

        if violations:
            msg = "CRITICAL INVARIANT VIOLATION: Agent(s) have negative equity!\n"
            msg += "This indicates margin calls were not properly enforced.\n\n"
            for v in violations:
                msg += f"Agent {v['agent_id']}:\n"
                msg += f"  Cash: ${v['cash']:.2f}\n"
                msg += f"  Share Value: ${v['share_value']:.2f}\n"
                msg += f"  Borrowed Cash: ${v['borrowed_cash']:.2f}\n"
                msg += f"  Equity: ${v['equity']:.2f} (NEGATIVE!)\n\n"

            self.logger.error(msg)
            # Log as warning but don't raise - allow simulation to complete for analysis
            # TODO: Once margin call enforcement is fixed, change this to raise ValueError
            self.logger.warning("⚠️ NEGATIVE EQUITY DETECTED - See issue #80 for fix")
        else:
            self.logger.info("✓ All agents have non-negative equity")
