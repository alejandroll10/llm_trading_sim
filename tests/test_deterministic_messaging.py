from pathlib import Path
import sys

# add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import types
import logging


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

from agents.deterministic.hold_agent import HoldTrader
from services.messaging_service import MessagingService


def test_deterministic_agents_persistent_messaging_three_rounds():
    # ensure clean messaging state
    MessagingService.reset()

    agent1 = HoldTrader(agent_id="A1")
    agent2 = HoldTrader(agent_id="A2")
    market_state = {'price': 100}
    history = []

    # round 1: agents make decisions and broadcast
    agent1.make_decision(market_state, history, round_number=1)
    agent2.make_decision(market_state, history, round_number=1)

    round1_messages = MessagingService.get_messages(1)
    assert len(round1_messages) == 2
    assert {m['agent_id'] for m in round1_messages} == {"A1", "A2"}

    # round 2: agents retrieve previous round messages
    last_msgs_a1 = agent1.get_last_round_messages(round_number=2)
    last_msgs_a2 = agent2.get_last_round_messages(round_number=2)
    assert last_msgs_a1 == round1_messages
    assert last_msgs_a2 == round1_messages

    # round 2: agents act again and broadcast new messages
    agent1.make_decision(market_state, history, round_number=2)
    agent2.make_decision(market_state, history, round_number=2)
    round2_messages = MessagingService.get_messages(2)
    assert len(round2_messages) == 2
    assert {m['agent_id'] for m in round2_messages} == {"A1", "A2"}

    # round 3: agents retrieve previous round messages and full history
    last_msgs_a1_r3 = agent1.get_last_round_messages(round_number=3)
    last_msgs_a2_r3 = agent2.get_last_round_messages(round_number=3)
    assert last_msgs_a1_r3 == round2_messages
    assert last_msgs_a2_r3 == round2_messages

    history_a1 = agent1.get_message_history(round_number=3)
    history_a2 = agent2.get_message_history(round_number=3)
    assert history_a1 == round1_messages + round2_messages
    assert history_a2 == round1_messages + round2_messages

    # round 3: agents act again and broadcast new messages
    agent1.make_decision(market_state, history, round_number=3)
    agent2.make_decision(market_state, history, round_number=3)
    round3_messages = MessagingService.get_messages(3)
    assert len(round3_messages) == 2
    assert {m['agent_id'] for m in round3_messages} == {"A1", "A2"}

    # message history should now contain all three rounds
    full_history_a1 = agent1.get_message_history(round_number=4)
    full_history_a2 = agent2.get_message_history(round_number=4)
    assert full_history_a1 == round1_messages + round2_messages + round3_messages
    assert full_history_a2 == round1_messages + round2_messages + round3_messages
