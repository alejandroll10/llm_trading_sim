import random
from dataclasses import dataclass
from agents.agent_manager.services.payment_services import PaymentDestination
from typing import Optional
from services.logging_service import LoggingService

@dataclass
class DividendPaymentResult:
    success: bool
    message: str
    total_payment: float
    num_shares_paid: int

@dataclass
class DividendInfo:
    """Immutable dividend model information"""
    base_dividend: float
    dividend_variation: float
    dividend_probability: float
    dividend_frequency: int
    expected_dividend: float
    max_dividend: float
    min_dividend: float
    destination: PaymentDestination
    # Factor loadings for shock structure (defaults for backwards compatibility)
    systematic_beta: float = 1.0
    style_gamma: float = 1.0

@dataclass
class DividendRealization:
    """Record of a dividend payment including shock components"""
    total_dividend: float
    base_component: float
    idiosyncratic_component: float
    systematic_shock: float = 0.0
    style_shock: float = 0.0
    systematic_contribution: float = 0.0  # beta * systematic_shock
    style_contribution: float = 0.0  # gamma * style_shock
class DividendCalculator:
    """Pure calculation logic - no state"""
    def __init__(self, dividend_params: dict):
        if not dividend_params:
            raise ValueError("dividend_params is required")
        self.model = dividend_params

    def calculate_dividend(self, systematic_shock: float = 0.0, style_shock: float = 0.0) -> DividendRealization:
        """Calculate actual dividend payment with optional factor shocks.

        Dividend structure:
            dividend = base + beta * systematic_shock + gamma * style_shock + idiosyncratic

        Args:
            systematic_shock: Market-wide shock affecting all stocks (drawn once per round)
            style_shock: Style-level shock affecting stocks in same category (drawn once per style per round)

        Returns:
            DividendRealization with total dividend and component breakdown
        """
        base = self.model['base_dividend']
        variation = self.model['dividend_variation']
        prob = self.model['dividend_probability']

        # Factor loadings (default to 1.0 for backwards compatibility)
        beta = self.model.get('systematic_beta', 1.0)
        gamma = self.model.get('style_gamma', 1.0)

        # Idiosyncratic component (existing stochastic logic)
        if random.random() < prob:
            idio = variation
        else:
            idio = -variation

        # Calculate contributions from each factor
        systematic_contribution = beta * systematic_shock
        style_contribution = gamma * style_shock

        # Total dividend
        total = base + systematic_contribution + style_contribution + idio

        return DividendRealization(
            total_dividend=total,
            base_component=base,
            idiosyncratic_component=idio,
            systematic_shock=systematic_shock,
            style_shock=style_shock,
            systematic_contribution=systematic_contribution,
            style_contribution=style_contribution
        )

    def should_pay_dividends(self, round_number: int) -> bool:
        """Pure function to determine payment rounds"""
        return round_number % self.model['dividend_frequency'] == 0

    def get_model_info(self) -> DividendInfo:
        """Get static model information"""
        base = self.model['base_dividend']
        variation = self.model['dividend_variation']
        prob = self.model['dividend_probability']

        expected = prob * (base + variation) + (1 - prob) * (base - variation)
        destination = self.model['destination']
        return DividendInfo(
            base_dividend=base,
            dividend_variation=variation,
            dividend_probability=prob,
            dividend_frequency=self.model['dividend_frequency'],
            expected_dividend=expected,
            max_dividend=base + variation,
            min_dividend=base - variation,
            destination=destination,
            systematic_beta=self.model.get('systematic_beta', 1.0),
            style_gamma=self.model.get('style_gamma', 1.0)
        )

class DividendPaymentProcessor:
    """Handles all payment operations"""
    def __init__(self, agent_repository, logger, stock_id="DEFAULT_STOCK"):
        self.agent_repository = agent_repository
        self.stock_id = stock_id  # Which stock this dividend service is for

    def _process_dividend_payment(self, dividend: float, destination: PaymentDestination, round_number: int) -> DividendPaymentResult:
        """Process actual payments"""
        total_shares = 0
        total_payment = 0

        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            state = self.agent_repository.get_agent_state_snapshot(
                agent_id, self.agent_repository.context.current_price
            )

            # Get shares for THIS specific stock only (multi-stock support)
            shares_in_stock = agent.positions.get(self.stock_id, 0)
            committed_shares_in_stock = agent.committed_positions.get(self.stock_id, 0)
            borrowed_shares_in_stock = agent.borrowed_positions.get(self.stock_id, 0)

            # Net share position accounts for short holdings
            net_position = shares_in_stock + committed_shares_in_stock - borrowed_shares_in_stock
            if net_position == 0:
                continue

            payment = dividend * net_position
            total_shares += abs(net_position)
            total_payment += payment

            account_type = "dividend" if destination == PaymentDestination.DIVIDEND_ACCOUNT else "main"
            self.agent_repository.update_account_balance(
                agent_id=agent_id,
                amount=payment,
                account_type=account_type,
                payment_type="dividend",
                round_number=round_number,
                stock_id=self.stock_id
            )

            action = "Paid" if payment >= 0 else "Deducted"
            LoggingService.get_logger('dividend').info(
                f"{action} {abs(payment):.2f} {'to' if payment >= 0 else 'from'} agent {agent_id} ({net_position} shares @ {dividend:.2f})"
            )
        
        LoggingService.get_logger('dividend').info(
            f"\n=== Round {round_number} Dividend Payment ===\n"
            f"Rate: ${dividend:.2f}\n"
            f"Total Shares: {total_shares}\n"
            f"Total Payment: ${total_payment:.2f}"
        )
        
        return DividendPaymentResult(
            success=True,
            message="Dividend payment processed successfully",
            total_payment=total_payment,
            num_shares_paid=total_shares
        )

    def process_redemption(self, redemption_value: float, round_number: int) -> DividendPaymentResult:
        """Process final redemption payment for this stock"""
        LoggingService.get_logger('dividend').info(f"\n=== Final Round {round_number} Redemption Payment for {self.stock_id} ===")
        LoggingService.get_logger('dividend').info(f"Redeeming {self.stock_id} shares at ${redemption_value:.2f} per share")

        total_shares = 0
        total_payment = 0

        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)

            # Get shares for THIS specific stock only (multi-stock support)
            # NOTE: committed_shares should be 0 since all orders are cancelled before redemption
            shares_in_stock = agent.positions.get(self.stock_id, 0)
            committed_shares_in_stock = agent.committed_positions.get(self.stock_id, 0)
            borrowed_shares_in_stock = agent.borrowed_positions.get(self.stock_id, 0)

            # All orders should be cancelled before redemption, so committed should be 0
            # If not, log a warning
            if committed_shares_in_stock != 0:
                LoggingService.get_logger('dividend').warning(
                    f"Agent {agent_id} has committed shares ({committed_shares_in_stock}) during redemption - "
                    f"orders should have been cancelled first!"
                )

            net_position = shares_in_stock + committed_shares_in_stock - borrowed_shares_in_stock
            payment = 0

            if net_position != 0:
                payment = redemption_value * net_position
                total_shares += abs(net_position)
                total_payment += payment

                self.agent_repository.update_account_balance(
                    agent_id=agent_id,
                    amount=payment,
                    account_type="main",
                    payment_type="redemption",
                    round_number=round_number,
                    stock_id=self.stock_id
                )

                action = "Redeemed" if payment >= 0 else "Covered"
                LoggingService.get_logger('dividend').info(
                    f"{action} {abs(net_position)} {self.stock_id} shares for agent {agent_id} at ${redemption_value:.2f}"
                )

            # Clear shares for THIS stock only
            # Redemption pays for ALL shares (positions + committed - borrowed), so clear everything
            agent._update_position(self.stock_id, 0)
            agent.committed_positions[self.stock_id] = 0  # We just paid for these!

            # Release borrowed shares back to the pool before clearing
            if self.stock_id in agent.borrowed_positions and agent.borrowed_positions[self.stock_id] > 0:
                borrowed = agent.borrowed_positions[self.stock_id]
                agent.borrowed_positions[self.stock_id] = 0
                # Use per-stock borrowing repo for multi-stock mode
                borrowing_repo = self.agent_repository._get_borrowing_repo(self.stock_id)
                borrowing_repo.release_shares(agent_id, borrowed)

        # After all positions are cleared, ensure aggregate short interest
        # reflects the forced covering that occurred during redemption.
        total_borrowed = sum(
            self.agent_repository.get_agent_state_snapshot(
                agent_id,
                self.agent_repository.context.current_price
            ).borrowed_shares
            for agent_id in self.agent_repository.get_all_agent_ids()
        )
        self.agent_repository.context.update_short_interest(total_borrowed)

        return DividendPaymentResult(
            success=True,
            message="Redemption payment processed successfully",
            total_payment=total_payment,
            num_shares_paid=total_shares
        )

class DividendService:
    """Coordinates dividend operations"""
    def __init__(self, agent_repository, logger, dividend_params, redemption_value=None, stock_id="DEFAULT_STOCK"):
        self.calculator = DividendCalculator(dividend_params)
        self.payment_processor = DividendPaymentProcessor(agent_repository, logger, stock_id)
        self.stock_id = stock_id
        self.dividend_history = []  # List of DividendRealization objects
        self.redemption_value = redemption_value
        self.current_round = 0
        self._should_pay_this_round = False
        # Store style for this stock (used for shock lookups)
        self.style = dividend_params.get('style', None)

        try:
            self.dividend_destination = PaymentDestination.from_string(
                dividend_params.get('destination', 'dividend')
            )
        except ValueError as e:
            LoggingService.get_logger('dividend').warning(f"{e}, defaulting to dividend account")
            self.dividend_destination = PaymentDestination.DIVIDEND_ACCOUNT

    def process_dividend_payments(
        self,
        round_number: int,
        systematic_shock: float = 0.0,
        style_shock: float = 0.0
    ) -> DividendPaymentResult:
        """Process dividend payments for the current round.

        Args:
            round_number: Current simulation round
            systematic_shock: Market-wide shock (same for all stocks)
            style_shock: Style-level shock (same for stocks in same style)

        Returns:
            DividendPaymentResult with payment details
        """
        realization = self.calculator.calculate_dividend(
            systematic_shock=systematic_shock,
            style_shock=style_shock
        )
        self.dividend_history.append(realization)

        LoggingService.get_logger('dividend').info(
            f"Dividend for {self.stock_id}: {realization.total_dividend:.4f} "
            f"(base={realization.base_component:.2f}, idio={realization.idiosyncratic_component:.2f}, "
            f"sys={realization.systematic_contribution:.4f}, style={realization.style_contribution:.4f})"
        )

        return self.payment_processor._process_dividend_payment(
            realization.total_dividend,
            self.dividend_destination,
            round_number
        )

    def process_round_end(
        self,
        round_number: int,
        is_final_round: bool,
        systematic_shock: float = 0.0,
        style_shock: float = 0.0
    ) -> Optional[DividendPaymentResult]:
        """Handle end-of-round processing.

        Args:
            round_number: Current simulation round
            is_final_round: Whether this is the final round (triggers redemption)
            systematic_shock: Market-wide shock for dividend calculation
            style_shock: Style-level shock for dividend calculation

        Returns:
            DividendPaymentResult if payments were made, None otherwise
        """
        self.current_round = round_number
        LoggingService.get_logger('dividend').info(f"Processing round end {round_number}")

        if self.calculator.should_pay_dividends(round_number) and not is_final_round:
            LoggingService.get_logger('dividend').info(f"Processing dividend payment for round {round_number}")
            return self.process_dividend_payments(
                round_number,
                systematic_shock=systematic_shock,
                style_shock=style_shock
            )

        if is_final_round and self.redemption_value is not None:
            LoggingService.get_logger('dividend').info(f"Processing redemption for final round {round_number}")
            return self.payment_processor.process_redemption(
                redemption_value=self.redemption_value,
                round_number=round_number
            )

        LoggingService.get_logger('dividend').info(f"No dividend or redemption for round {round_number}")
        return None

    def _get_next_payment_round(self) -> int:
        """Calculate next payment round"""
        frequency = self.calculator.model['dividend_frequency']
        return frequency - (self.current_round % frequency)

    def update(self, round_number: int):
        """Update state for the current round"""
        self.current_round = round_number
        self._should_pay_this_round = self.calculator.should_pay_dividends(round_number)

    def get_last_realization(self) -> Optional[DividendRealization]:
        """Get the most recent dividend realization with all components."""
        return self.dividend_history[-1] if self.dividend_history else None

    def get_state(self) -> dict:
        """Single source of truth for dividend state"""
        model_info = self.calculator.get_model_info()
        last_realization = self.get_last_realization()

        # For backwards compatibility, extract total_dividend as float
        last_paid = last_realization.total_dividend if last_realization else None

        return {
            'model': model_info,
            'last_paid_dividend': last_paid,
            'last_realization': last_realization,  # Full breakdown for new consumers
            'redemption_value': self.redemption_value,
            'next_payment_round': self._get_next_payment_round(),
            'should_pay': self._should_pay_this_round,
            'style': self.style
        }