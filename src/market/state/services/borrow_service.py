from dataclasses import dataclass
from services.logging_service import LoggingService


@dataclass
class BorrowFeeResult:
    success: bool
    message: str
    total_fee: float
    num_accounts_charged: int


class BorrowService:
    """Service to handle borrow fee calculations and payments"""

    def __init__(self, agent_repository, logger, borrow_params):
        self.agent_repository = agent_repository

        if not borrow_params:
            raise ValueError("borrow_params is required")

        self.borrow_model = borrow_params
        self.borrow_history = []
        self.current_round = 0
        self.next_payment_round = self.borrow_model.get('payment_frequency', 1)

    def calculate_fee(self, borrowed_shares: float, price: float) -> float:
        """Calculate borrow fee for a position"""
        return borrowed_shares * self.borrow_model['rate'] * price

    def process_borrow_fees(self, round_number: int, price: float) -> BorrowFeeResult:
        """Process borrow fee payments for all agents"""
        frequency = self.borrow_model.get('payment_frequency', 1)
        if round_number % frequency != 0:
            return BorrowFeeResult(True, "No borrow fees this round", 0.0, 0)

        total_fee = 0.0
        num_agents = 0

        for agent_id in self.agent_repository.get_all_agent_ids():
            agent = self.agent_repository.get_agent(agent_id)
            if agent.borrowed_shares > 0:
                fee = self.calculate_fee(agent.borrowed_shares, price)
                if fee > 0:
                    self.agent_repository.update_account_balance(
                        agent_id=agent_id,
                        amount=-fee,
                        account_type='main',
                        payment_type='borrow_fee',
                        round_number=round_number
                    )
                    total_fee += fee
                    num_agents += 1

        self.borrow_history.append(total_fee)
        LoggingService.get_logger('borrow').info(
            f"\n=== Round {round_number} Borrow Fees ==="\
            f"\nRate: {self.borrow_model['rate']:.1%}"\
            f"\nTotal Fees: ${total_fee:.2f}"
        )

        return BorrowFeeResult(
            success=True,
            message="Borrow fees processed successfully",
            total_fee=total_fee,
            num_accounts_charged=num_agents
        )

    def get_current_rate(self) -> float:
        """Get current borrow fee rate"""
        return self.borrow_model['rate']

    def get_state(self) -> dict:
        """Get current borrow fee state"""
        return {
            'rate': self.get_current_rate(),
            'payment_frequency': self.borrow_model.get('payment_frequency', 1),
            'last_payment': self.borrow_history[-1] if self.borrow_history else None,
            'next_payment_round': self.next_payment_round
        }

    def update(self, round_number: int):
        """Update state for current round"""
        self.current_round = round_number
        frequency = self.borrow_model.get('payment_frequency', 1)
        self.next_payment_round = (
            round_number + (frequency - (round_number % frequency))
            if round_number % frequency != 0
            else round_number + frequency
        )
