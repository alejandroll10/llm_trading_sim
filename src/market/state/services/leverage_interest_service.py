from typing import List, Dict
from agents.base_agent import BaseAgent
from services.logging_service import LoggingService


class LeverageInterestService:
    """Charges interest on borrowed cash for leveraged positions.

    This service processes interest charges each round for agents who have
    borrowed cash to take leveraged long positions. Interest is calculated
    based on the amount borrowed and the per-round interest rate.
    """

    def __init__(self, interest_rate: float = 0.05):
        """Initialize the leverage interest service.

        Args:
            interest_rate: Per-round interest rate on borrowed cash (e.g., 0.05 = 5% per round)
        """
        self.interest_rate = interest_rate
        self.total_interest_charged = 0.0
        self.logger = LoggingService.get_logger('leverage_interest')

        # Handle case where LoggingService returns None
        if self.logger is None:
            import logging
            self.logger = logging.getLogger('leverage_interest')
            self.logger.setLevel(logging.INFO)

        self.logger.info(
            f"Initialized leverage interest service with {interest_rate:.2%} per-round rate"
        )

    def charge_interest(
        self,
        agents: List[BaseAgent]
    ) -> Dict[str, float]:
        """Charge per-round interest on borrowed cash for all agents.

        Interest is calculated as: borrowed_cash * interest_rate
        The interest is deducted from the agent's cash and added to their
        cumulative leverage_interest_paid tracker.

        Args:
            agents: List of all agents in the simulation

        Returns:
            Dict mapping agent_id to interest charged this round
        """
        interest_by_agent = {}

        for agent in agents:
            if agent.borrowed_cash > 0:
                # Calculate interest for this round
                interest = agent.borrowed_cash * self.interest_rate

                # Charge interest (reduce cash)
                agent.cash -= interest
                agent.leverage_interest_paid += interest

                interest_by_agent[agent.agent_id] = interest
                self.total_interest_charged += interest

                # Log interest charge
                agent.record_payment(
                    account='main',
                    amount=-interest,  # Negative = outflow
                    payment_type='interest',
                    round_number=agent.last_update_round
                )

                self.logger.debug(
                    f"Charged {agent.agent_id} ${interest:.2f} interest "
                    f"on ${agent.borrowed_cash:.2f} borrowed cash "
                    f"(rate: {self.interest_rate:.4%}/round)"
                )

        if interest_by_agent:
            total_round_interest = sum(interest_by_agent.values())
            self.logger.info(
                f"Charged ${total_round_interest:.2f} total interest to "
                f"{len(interest_by_agent)} agents with borrowed cash"
            )

        return interest_by_agent

    def get_total_interest_charged(self) -> float:
        """Get cumulative interest charged across all rounds.

        Returns:
            Total interest charged since service initialization
        """
        return self.total_interest_charged

    def reset_statistics(self):
        """Reset cumulative statistics (useful for new simulations)."""
        self.total_interest_charged = 0.0
        self.logger.info("Reset leverage interest service statistics")
