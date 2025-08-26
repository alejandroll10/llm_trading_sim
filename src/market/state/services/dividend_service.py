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
class DividendCalculator:
    """Pure calculation logic - no state"""
    def __init__(self, dividend_params: dict):
        if not dividend_params:
            raise ValueError("dividend_params is required")
        self.model = dividend_params
        
    def calculate_dividend(self) -> float:
        """Calculate actual dividend payment"""
        if random.random() < self.model['dividend_probability']:
            return self.model['base_dividend'] + self.model['dividend_variation']
        return self.model['base_dividend'] - self.model['dividend_variation']

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
            destination=destination
        )

class DividendPaymentProcessor:
    """Handles all payment operations"""
    def __init__(self, agent_repository, logger):
        self.agent_repository = agent_repository

    def _process_dividend_payment(self, dividend: float, destination: PaymentDestination, round_number: int) -> DividendPaymentResult:
        """Process actual payments"""
        total_shares = 0
        total_payment = 0

        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            state = self.agent_repository.get_agent_state_snapshot(
                agent_id, self.agent_repository.context.current_price
            )

            # Net share position accounts for short holdings
            net_position = agent.total_shares - agent.borrowed_shares
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
                round_number=round_number
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
        """Process final redemption payment"""
        LoggingService.get_logger('dividend').info(f"\n=== Final Round {round_number} Redemption Payment ===")
        LoggingService.get_logger('dividend').info(f"Redeeming all shares at ${redemption_value:.2f} per share")
        
        total_shares = 0
        total_payment = 0
        
        for agent_id in self.agent_repository.get_all_agent_ids():
            state = self.agent_repository.get_agent_state_snapshot(
                agent_id, self.agent_repository.context.current_price
            )

            net_position = state.net_shares
            payment = 0

            if net_position != 0:
                payment = redemption_value * net_position
                total_shares += abs(net_position)
                total_payment += payment

                self.agent_repository.update_account_balance(
                    agent_id=agent_id,
                    amount=payment,
                    account_type="main"
                )

                action = "Redeemed" if payment >= 0 else "Covered"
                LoggingService.get_logger('dividend').info(
                    f"{action} {abs(net_position)} shares for agent {agent_id} at ${redemption_value:.2f}"
                )

            # Always clear shares and outstanding borrows
            self.agent_repository.redeem_all_shares(agent_id)

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
    def __init__(self, agent_repository, logger, dividend_params, redemption_value=None):
        self.calculator = DividendCalculator(dividend_params)
        self.payment_processor = DividendPaymentProcessor(agent_repository, logger)
        self.dividend_history = []
        self.redemption_value = redemption_value
        self.current_round = 0
        self._should_pay_this_round = False
        
        try:
            self.dividend_destination = PaymentDestination.from_string(
                dividend_params.get('destination', 'dividend')
            )
        except ValueError as e:
            LoggingService.get_logger('dividend').warning(f"{e}, defaulting to dividend account")
            self.dividend_destination = PaymentDestination.DIVIDEND_ACCOUNT

    def process_dividend_payments(self, round_number: int) -> DividendPaymentResult:
        """Process dividend payments for the current round"""
        dividend = self.calculator.calculate_dividend()
        self.dividend_history.append(dividend)
        return self.payment_processor._process_dividend_payment(
            dividend, 
            self.dividend_destination, 
            round_number
        )

    def process_round_end(self, round_number: int, is_final_round: bool) -> Optional[DividendPaymentResult]:
        """Handle end-of-round processing"""

        self.current_round = round_number  # Update round number
        LoggingService.get_logger('dividend').info(f"Processing round end {round_number}")

        if self.calculator.should_pay_dividends(round_number) and not is_final_round:
            LoggingService.get_logger('dividend').info(f"Processing dividend payment for round {round_number}")
            return self.process_dividend_payments(round_number)
        
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

    def get_state(self) -> dict:
        """Single source of truth for dividend state"""
        model_info = self.calculator.get_model_info()
        last_paid = self.dividend_history[-1] if self.dividend_history else None
        
        return {
            'model': model_info,
            'last_paid_dividend': last_paid,
            'redemption_value': self.redemption_value,
            'next_payment_round': self._get_next_payment_round(),
            'should_pay': self._should_pay_this_round
        }