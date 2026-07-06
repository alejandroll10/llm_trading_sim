"""Tests for the MessagingService feed-transform hook (issue #95).

Writers always store raw messages; readers see the transformed feed.
Raw messages (with sentiment scores) are always exported for logging.
"""

from pathlib import Path
import sys

import pytest

# add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from services.messaging_service import MessagingService
from services.sentiment_scorer import score_sentiment, sentiment_label


@pytest.fixture(autouse=True)
def clean_messaging():
    MessagingService.reset()
    yield
    MessagingService.reset()


def _post_round_one():
    MessagingService.add_message(1, "A1", "Time to buy, this stock is undervalued and strong!")
    MessagingService.add_message(1, "A2", "Sell now, overvalued bubble about to crash.")
    MessagingService.add_message(1, "A3", "The weather is nice today.")


class TestSentimentScorer:
    def test_bullish_message_scores_positive(self):
        assert score_sentiment("buy the rally, strong upside") > 0.25

    def test_bearish_message_scores_negative(self):
        assert score_sentiment("sell this bubble before the crash") < -0.25

    def test_neutral_message_scores_zero(self):
        assert score_sentiment("the weather is nice today") == 0.0

    def test_dict_messages_are_flattened(self):
        assert score_sentiment({"text": "buy buy buy"}) > 0

    def test_label_format(self):
        assert sentiment_label(0.7) == "bullish (+0.70)"
        assert sentiment_label(-0.7) == "bearish (-0.70)"
        assert sentiment_label(0.0) == "neutral (+0.00)"


class TestIdentityTransform:
    def test_default_is_identity(self):
        _post_round_one()
        msgs = MessagingService.get_messages(1)
        assert [m["agent_id"] for m in msgs] == ["A1", "A2", "A3"]
        assert msgs[0]["message"].startswith("Time to buy")

    def test_reader_view_has_no_sentiment_fields(self):
        _post_round_one()
        for m in MessagingService.get_messages(1):
            assert set(m.keys()) == {"agent_id", "message"}


class TestSentimentOnlyTransform:
    def test_readers_see_only_sentiment(self):
        MessagingService.configure(transform="sentiment_only")
        _post_round_one()
        msgs = MessagingService.get_messages(1)
        assert msgs[0]["message"].startswith("bullish")
        assert msgs[1]["message"].startswith("bearish")
        assert msgs[2]["message"].startswith("neutral")
        # authorship preserved
        assert [m["agent_id"] for m in msgs] == ["A1", "A2", "A3"]

    def test_raw_export_unchanged(self):
        MessagingService.configure(transform="sentiment_only")
        _post_round_one()
        raw = MessagingService.get_all_messages()
        assert raw[0]["message"].startswith("Time to buy")
        assert raw[0]["sentiment_label"].startswith("bullish")
        assert raw[1]["sentiment_score"] < 0


class TestScrambledTransform:
    def test_content_preserved_authors_permuted(self):
        MessagingService.configure(transform="scrambled", seed=42)
        _post_round_one()
        msgs = MessagingService.get_messages(1)
        # same content mix, same author set, message order unchanged
        assert [m["message"] for m in msgs] == [
            m["message"] for m in MessagingService.get_all_messages()
        ]
        assert {m["agent_id"] for m in msgs} == {"A1", "A2", "A3"}

    def test_deterministic_across_reads_and_runs(self):
        MessagingService.configure(transform="scrambled", seed=42)
        _post_round_one()
        first = MessagingService.get_messages(1)
        second = MessagingService.get_messages(1)
        assert first == second  # every reader sees the same scramble

        # matched-seed reproducibility after a reset
        MessagingService.reset()
        MessagingService.configure(transform="scrambled", seed=42)
        _post_round_one()
        assert MessagingService.get_messages(1) == first

    def test_different_seed_can_differ(self):
        MessagingService.configure(transform="scrambled", seed=42)
        _post_round_one()
        assignment_42 = [m["agent_id"] for m in MessagingService.get_messages(1)]

        MessagingService.reset()
        MessagingService.configure(transform="scrambled", seed=7)
        _post_round_one()
        assignment_7 = [m["agent_id"] for m in MessagingService.get_messages(1)]

        # Not guaranteed to differ for any single seed pair, but these do;
        # the invariant that matters is each is internally deterministic.
        assert assignment_42 != assignment_7 or len(assignment_42) == 1


class TestMutedTransform:
    def test_readers_see_nothing(self):
        MessagingService.configure(transform="muted")
        _post_round_one()
        assert MessagingService.get_messages(1) == []
        assert MessagingService.get_message_history(1) == []

    def test_raw_messages_still_logged(self):
        MessagingService.configure(transform="muted")
        _post_round_one()
        assert len(MessagingService.get_all_messages()) == 3


class TestConfiguration:
    def test_invalid_transform_rejected(self):
        with pytest.raises(ValueError, match="Unknown feed transform"):
            MessagingService.configure(transform="loud")

    def test_reset_restores_identity(self):
        MessagingService.configure(transform="muted")
        MessagingService.reset()
        _post_round_one()
        assert len(MessagingService.get_messages(1)) == 3

    def test_history_uses_transform(self):
        MessagingService.configure(transform="sentiment_only")
        _post_round_one()
        MessagingService.add_message(2, "A1", "still bullish, buy more")
        history = MessagingService.get_message_history(2)
        assert len(history) == 4
        assert all(
            m["message"].split(" ")[0] in {"bullish", "bearish", "neutral"}
            for m in history
        )

    def test_empty_message_ignored(self):
        MessagingService.add_message(1, "A1", "")
        assert MessagingService.get_messages(1) == []
