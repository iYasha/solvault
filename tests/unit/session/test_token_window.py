from datetime import UTC, datetime

from sol.session.models import ChatMessage
from sol.session.token_window import apply_token_window, count_tokens


def _make_message(content: str, token_count: int = 0) -> ChatMessage:
    """Create a ChatMessage instance for testing without DB."""
    return ChatMessage(
        id="test",
        session_id="test-session",
        role="user",
        content=content,
        timestamp=datetime.now(UTC),
        token_count=token_count,
    )


class TestCountTokens:
    """Token counting should work for known and unknown models."""

    def test_counts_tokens_for_known_model(self):
        """Given a known model, when counting tokens, then a positive count is returned."""
        count = count_tokens("hello world", model="gpt-4")
        assert count > 0, "Token count should be positive for non-empty text"

    def test_falls_back_for_unknown_model(self):
        """Given an unknown model, when counting tokens, then cl100k_base fallback is used."""
        count = count_tokens("hello world", model="unknown-model-xyz")
        assert count > 0, "Fallback encoding should still count tokens"

    def test_empty_string_returns_zero(self):
        """Given an empty string, when counting tokens, then zero is returned."""
        count = count_tokens("", model="gpt-4")
        assert count == 0, "Empty string should have zero tokens"


class TestApplyTokenWindow:
    """Token windowing should return the most recent messages within budget."""

    def test_returns_all_messages_within_budget(self):
        """Given messages totaling under max_tokens, when windowed, then all are returned."""
        messages = [
            _make_message("hello", token_count=5),
            _make_message("world", token_count=5),
        ]
        result = apply_token_window(messages, max_tokens=100)
        assert len(result) == 2, "All messages should fit within budget"

    def test_truncates_oldest_messages(self):
        """Given messages exceeding budget, when windowed, then oldest are dropped."""
        messages = [
            _make_message("old message", token_count=50),
            _make_message("middle message", token_count=50),
            _make_message("recent message", token_count=50),
        ]
        result = apply_token_window(messages, max_tokens=80)
        assert len(result) == 1, "Only the most recent message should fit"
        assert result[0].content == "recent message", "Most recent message should be kept"

    def test_always_includes_most_recent_message(self):
        """Given a single message exceeding budget, when windowed, then it's still included."""
        messages = [_make_message("very long message", token_count=1000)]
        result = apply_token_window(messages, max_tokens=10)
        assert len(result) == 1, "Most recent message should always be included regardless of size"

    def test_empty_messages_returns_empty(self):
        """Given no messages, when windowed, then empty list is returned."""
        result = apply_token_window([], max_tokens=100)
        assert result == [], "Empty input should return empty output"

    def test_preserves_chronological_order(self):
        """Given messages within budget, when windowed, then chronological order is preserved."""
        messages = [
            _make_message("first", token_count=10),
            _make_message("second", token_count=10),
            _make_message("third", token_count=10),
        ]
        result = apply_token_window(messages, max_tokens=100)
        assert [m.content for m in result] == ["first", "second", "third"], "Messages should be in chronological order"
