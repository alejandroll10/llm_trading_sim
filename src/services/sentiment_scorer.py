"""Lexicon-based sentiment scoring for agent social messages.

Used by the MessagingService feed-transform hook (issue #95) to support the
``sentiment_only`` counterfactual feed, where readers see only a coarse
sentiment label instead of the raw message text. Scores are also logged
alongside raw messages in every mode, so they are available as data even for
``identity`` runs.

The scorer is deliberately deterministic and free (no API calls). If a more
accurate scorer is needed, replace ``score_sentiment`` with an LLM-backed
implementation keeping the same [-1, 1] contract.
"""

import re
from typing import Any

BULLISH_WORDS = {
    'buy', 'buying', 'bull', 'bullish', 'up', 'upside', 'rally', 'rallying',
    'rise', 'rising', 'gain', 'gains', 'strong', 'strength', 'long',
    'undervalued', 'cheap', 'bargain', 'growth', 'breakout', 'moon',
    'optimistic', 'surge', 'surging', 'accumulate', 'accumulating', 'hold',
    'holding', 'opportunity', 'momentum', 'higher', 'winner', 'profit',
    'profits', 'confident', 'positive',
}

BEARISH_WORDS = {
    'sell', 'selling', 'bear', 'bearish', 'down', 'downside', 'crash',
    'crashing', 'drop', 'dropping', 'fall', 'falling', 'decline', 'declining',
    'weak', 'weakness', 'short', 'shorting', 'overvalued', 'expensive',
    'bubble', 'dump', 'dumping', 'correction', 'fear', 'panic', 'pessimistic',
    'lower', 'loser', 'loss', 'losses', 'risky', 'negative', 'exit', 'avoid',
}

_WORD_RE = re.compile(r"[a-z']+")


def message_to_text(message: Any) -> str:
    """Flatten a message payload (string or structured dict) to plain text."""
    if isinstance(message, dict):
        return " ".join(str(v) for v in message.values())
    return str(message)


def score_sentiment(message: Any) -> float:
    """Score a message's sentiment in [-1, 1] (-1 bearish, +1 bullish)."""
    words = _WORD_RE.findall(message_to_text(message).lower())
    bullish = sum(1 for w in words if w in BULLISH_WORDS)
    bearish = sum(1 for w in words if w in BEARISH_WORDS)
    total = bullish + bearish
    if total == 0:
        return 0.0
    return (bullish - bearish) / total


def sentiment_label(score: float) -> str:
    """Coarse label + magnitude, e.g. 'bullish (+0.70)'."""
    if score >= 0.25:
        direction = "bullish"
    elif score <= -0.25:
        direction = "bearish"
    else:
        direction = "neutral"
    return f"{direction} ({score:+.2f})"
