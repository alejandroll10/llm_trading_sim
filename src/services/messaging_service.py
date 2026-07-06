import random
from typing import Any, Dict, List

from services.sentiment_scorer import score_sentiment, sentiment_label


class MessagingService:
    """Simple in-memory broadcast channel for agent messages.

    Messages accumulate during simulation and are cleared on reset.
    For very long simulations (100+ rounds), consider memory implications.

    Feed transforms (issue #95): writers always store the raw message; readers
    see a transformed version according to the configured feed transform. Raw
    messages (with sentiment scores) are always available via
    ``get_all_messages`` for logging to social_messages.csv.

    Transforms:
    - ``identity``: readers see raw messages (default).
    - ``sentiment_only``: each message is replaced by its coarse scored
      sentiment, e.g. "bullish (+0.70)".
    - ``scrambled``: messages within a round are randomly reassigned to
      authors (deterministic per seed/round) — breaks source credibility
      while preserving the content mix.
    - ``muted``: readers see no messages at all.
    """

    VALID_TRANSFORMS = ('identity', 'sentiment_only', 'scrambled', 'muted')

    _messages: Dict[int, List[dict]] = {}
    _transform: str = 'identity'
    _seed: int = 0

    @classmethod
    def configure(cls, transform: str = 'identity', seed: int = 0) -> None:
        """Set the feed transform applied on the read path.

        Args:
            transform: One of ``VALID_TRANSFORMS``.
            seed: Seed for the deterministic per-round shuffle used by the
                ``scrambled`` transform (ignored by other transforms).
        """
        if transform not in cls.VALID_TRANSFORMS:
            raise ValueError(
                f"Unknown feed transform '{transform}'. "
                f"Valid options: {cls.VALID_TRANSFORMS}"
            )
        cls._transform = transform
        cls._seed = seed

    @classmethod
    def get_messages(cls, round_number: int) -> List[dict]:
        """Return the messages readers see for a given round (transformed)."""
        return cls._apply_transform(cls._messages.get(round_number, []), round_number)

    @classmethod
    def add_message(cls, round_number: int, agent_id: str, message: Dict[str, Any]) -> None:
        """Store a structured message for the specified round.

        The raw message is always stored; the sentiment score is computed at
        write time and logged alongside it (useful as data in any mode).
        """
        if not message:
            return
        score = score_sentiment(message)
        cls._messages.setdefault(round_number, []).append({
            "agent_id": agent_id,
            "message": message,
            "sentiment_score": round(score, 4),
            "sentiment_label": sentiment_label(score),
        })

    @classmethod
    def get_message_history(cls, up_to_round: int) -> List[dict]:
        """Return all messages from round 1 through ``up_to_round`` inclusive
        (transformed, as readers would have seen them)."""
        history: List[dict] = []
        for r in range(1, up_to_round + 1):
            history.extend(cls.get_messages(r))
        return history

    @classmethod
    def get_all_messages(cls) -> List[dict]:
        """Get all RAW messages from all rounds for export/logging.

        Not subject to the feed transform: even under counterfactual feeds,
        the original messages (plus sentiment scores) are what get logged.
        """
        all_messages = []
        for round_num in sorted(cls._messages.keys()):
            for msg_data in cls._messages[round_num]:
                all_messages.append({
                    'round': round_num,
                    'agent_id': msg_data['agent_id'],
                    'message': msg_data['message'],
                    'sentiment_score': msg_data['sentiment_score'],
                    'sentiment_label': msg_data['sentiment_label'],
                })
        return all_messages

    @classmethod
    def _apply_transform(cls, stored: List[dict], round_number: int) -> List[dict]:
        """Produce the reader-facing view of a round's messages."""
        if cls._transform == 'muted':
            return []
        if cls._transform == 'sentiment_only':
            return [
                {"agent_id": m["agent_id"], "message": m["sentiment_label"]}
                for m in stored
            ]
        if cls._transform == 'scrambled':
            # Reassign messages to authors with a shuffle that is
            # deterministic per (seed, round) so every reader of a round sees
            # the same scramble and matched-seed runs reproduce exactly.
            authors = [m["agent_id"] for m in stored]
            rng = random.Random(f"{cls._seed}:{round_number}")
            rng.shuffle(authors)
            return [
                {"agent_id": author, "message": m["message"]}
                for author, m in zip(authors, stored)
            ]
        # identity
        return [{"agent_id": m["agent_id"], "message": m["message"]} for m in stored]

    @classmethod
    def reset(cls) -> None:
        """Clear all stored messages and restore the default transform."""
        cls._messages.clear()
        cls._transform = 'identity'
        cls._seed = 0
