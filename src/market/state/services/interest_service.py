from dataclasses import dataclass
from agents.agent_manager.services.payment_services import PaymentDestination
from services.logging_service import LoggingService

@dataclass
class InterestPaymentResult:
    success: bool
    message: str
    total_payment: float
    num_accounts_paid: int

class InterestService:
    def __init__(self, agent_repository, logger, interest_params):
        """Initialize interest service with repository"""
        self.agent_repository = agent_repository
        
        if not interest_params:
            raise ValueError("interest_params is required")
            
        self.interest_model = interest_params
        
        # Add destination handling similar to dividend service
        try:
            self.interest_destination = PaymentDestination.from_string(
                self.interest_model.get('destination', 'dividend')  # Default to dividend account
            )
        except ValueError as e:
            LoggingService.get_logger('interest').warning(str(e))
            LoggingService.get_logger('interest').warning("Defaulting to dividend account")
            self.interest_destination = PaymentDestination.DIVIDEND_ACCOUNT
            
        self.interest_history = []
        self.current_round = 0
        self.next_payment_round = self.interest_model['compound_frequency']

    def calculate_interest(self, balance: float) -> float:
        """Calculate interest for a given balance"""
        return balance * self.interest_model['rate']

    def process_interest_payments(self, round_number: int) -> InterestPaymentResult:
        """Process interest payments through repository"""
        rate = self.interest_model['rate']
        total_interest = 0
        num_accounts_paid = 0
        
        for agent_id in self.agent_repository.get_all_agent_ids():
            # Get current balances
            balances = self.agent_repository.get_account_balances(agent_id)
            
            # Calculate interest on main and dividend accounts
            main_interest = self.calculate_interest(balances['main'])
            dividend_interest = self.calculate_interest(balances['dividend'])
            
            # Determine destination account based on configuration
            destination_account = (
                "dividend" if self.interest_destination == PaymentDestination.DIVIDEND_ACCOUNT 
                else "main"
            )
            
            # Update balances through repository
            if main_interest > 0:
                self.agent_repository.update_account_balance(
                    agent_id=agent_id,
                    amount=main_interest,
                    account_type=destination_account,
                    payment_type="interest",
                    round_number=round_number
                )
                total_interest += main_interest
                num_accounts_paid += 1
                
            if dividend_interest > 0:
                self.agent_repository.update_account_balance(
                    agent_id=agent_id,
                    amount=dividend_interest,
                    account_type=destination_account,
                    payment_type="interest",
                    round_number=round_number
                )
                total_interest += dividend_interest
                num_accounts_paid += 1
                
            if main_interest > 0 or dividend_interest > 0:
                LoggingService.get_logger('interest').info(
                    f"Paid interest to agent {agent_id}:"
                    f"\n - Main account: ${main_interest:.2f}"
                    f"\n - Dividend account: ${dividend_interest:.2f}"
                )
        
        LoggingService.get_logger('interest').info(
            f"\n=== Round {round_number} Interest Payments ==="
            f"\nRate: {rate:.1%}"
            f"\nTotal Interest: ${total_interest:.2f}"
        )
        
        return InterestPaymentResult(
            success=True,
            message="Interest payments processed successfully",
            total_payment=total_interest,
            num_accounts_paid=num_accounts_paid
        )

    def get_current_rate(self) -> float:
        """Get current interest rate"""
        return self.interest_model['rate']

    def get_state(self) -> dict:
        """Get current interest state"""
        return {
            'rate': self.get_current_rate(),
            'compound_frequency': self.interest_model['compound_frequency'],
            'last_payment': self.interest_history[-1] if self.interest_history else None,
            'next_payment_round': self.next_payment_round,
            'destination': self.interest_destination.value
        }

    def update(self, round_number: int):
        """Update state for current round"""
        self.current_round = round_number
        
        # Update next payment round
        frequency = self.interest_model['compound_frequency']
        self.next_payment_round = (
            round_number + (frequency - (round_number % frequency))
            if round_number % frequency != 0
            else round_number + frequency
        )