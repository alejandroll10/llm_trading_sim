from typing import Any, Dict, List


class MessagingService:
    """Simple in-memory broadcast channel for agent messages."""

    _messages: Dict[int, List[dict]] = {}

    @classmethod
    def get_messages(cls, round_number: int) -> List[dict]:
        """Return all messages for a given round."""
        return cls._messages.get(round_number, [])

    @classmethod
    def add_message(cls, round_number: int, agent_id: str, message: Dict[str, Any]) -> None:
        """Store a structured message for the specified round."""
        if not message:
            return
        cls._messages.setdefault(round_number, []).append({
            "agent_id": agent_id,
            "message": message,
        })

    @classmethod
    def get_message_history(cls, up_to_round: int) -> List[dict]:
        """Return all messages from round 1 through ``up_to_round`` inclusive."""
        history: List[dict] = []
        for r in range(1, up_to_round + 1):
            history.extend(cls.get_messages(r))
        return history

    @classmethod
    def reset(cls) -> None:
        """Clear all stored messages."""
        cls._messages = {}
