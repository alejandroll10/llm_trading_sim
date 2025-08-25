import sys
import types
import logging
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


class _TestLoggingService:
    @staticmethod
    def get_logger(name):
        return logging.getLogger(name)

    @staticmethod
    def log_agent_state(*args, **kwargs):
        pass

    @staticmethod
    def log_validation_error(*args, **kwargs):
        pass


sys.modules.setdefault("services.logging_service", types.ModuleType("services.logging_service"))
sys.modules["services.logging_service"].LoggingService = _TestLoggingService

from agents.base_agent import BaseAgent
from agents.agent_manager.agent_repository import AgentRepository
from agents.agents_api import TradeDecision
from market.state.services.dividend_service import DividendService
from market.state.sim_context import SimulationContext


class DummyAgent(BaseAgent):
    def make_decision(self, market_state, history, round_number):
        return TradeDecision(orders=[], replace_decision="Cancel", reasoning="")


def test_dividend_payments_long_flat_short():
    context = SimulationContext(
        num_rounds=1,
        initial_price=100,
        fundamental_price=100,
        redemption_value=0,
        transaction_cost=0,
    )
    long_agent = DummyAgent("long", initial_cash=0, initial_shares=10)
    flat_agent = DummyAgent("flat", initial_cash=0, initial_shares=0)
    short_agent = DummyAgent(
        "short", initial_cash=0, initial_shares=-5, allow_short_selling=True
    )
    repo = AgentRepository(
        [long_agent, flat_agent, short_agent], logger=None, context=context
    )

    dividend_params = {
        "base_dividend": 2.0,
        "dividend_variation": 0.0,
        "dividend_probability": 1.0,
        "dividend_frequency": 1,
        "destination": "dividend",
    }
    service = DividendService(repo, logger=None, dividend_params=dividend_params)

    service.process_dividend_payments(round_number=1)

    assert long_agent.dividend_cash == pytest.approx(20)
    assert flat_agent.dividend_cash == pytest.approx(0)
    assert short_agent.dividend_cash == pytest.approx(-10)

    assert long_agent.payment_history["dividend"][0].amount == pytest.approx(20)
    assert short_agent.payment_history["dividend"][0].amount == pytest.approx(-10)
    assert flat_agent.payment_history["dividend"] == []
