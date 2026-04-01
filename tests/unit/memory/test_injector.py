from sol.memory.injector import MemoryInjector
from sol.memory.retriever import RetrievalResult


def _make_result(content: str, memory_type: str = "facts", score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(content=content, memory_type=memory_type, score=score)


class TestBuildMemoryContext:
    """MemoryInjector should format retrieval results for system prompt injection."""

    def test_returns_empty_for_no_results(self):
        """Given no results, when build_memory_context called, then empty string returned."""
        injector = MemoryInjector()

        result = injector.build_memory_context([])

        assert result == "", "Should return empty string for no results"

    def test_formats_single_result(self):
        """Given one result, when build_memory_context called, then output contains type and content."""
        injector = MemoryInjector()
        results = [_make_result("User prefers Python", "user")]

        output = injector.build_memory_context(results)

        assert "[user]" in output, "Should contain memory type"
        assert "User prefers Python" in output, "Should contain memory content"
        assert "## Relevant Memories" in output, "Should contain header"

    def test_formats_multiple_results(self):
        """Given multiple results, when build_memory_context called, then all are included."""
        injector = MemoryInjector()
        results = [
            _make_result("Fact one", "facts"),
            _make_result("Fact two", "user"),
        ]

        output = injector.build_memory_context(results)

        assert "Fact one" in output, "Should contain first fact"
        assert "Fact two" in output, "Should contain second fact"

    def test_respects_max_tokens_budget(self):
        """Given results exceeding max_tokens, when called, then output is truncated."""
        injector = MemoryInjector()
        results = [
            _make_result("Short fact"),
            _make_result("A " * 5000),
        ]

        output = injector.build_memory_context(results, max_tokens=50)

        assert "Short fact" in output, "First result should fit"
        assert "A A A A" not in output, "Second result should be truncated"

    def test_returns_empty_if_no_results_fit(self):
        """Given all results exceed budget, when called, then empty string returned."""
        injector = MemoryInjector()
        results = [_make_result("x " * 5000)]

        output = injector.build_memory_context(results, max_tokens=10)

        assert output == "", "Should return empty when no results fit after header"
