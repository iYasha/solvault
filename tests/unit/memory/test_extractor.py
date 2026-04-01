import json
from unittest.mock import AsyncMock, MagicMock

from sol.memory.extractor import MemoryExtractor


def _make_mock_store():
    store = AsyncMock()
    store.build_manifest = AsyncMock(return_value="")
    store.find_similar = AsyncMock(return_value=None)  # no duplicates by default

    mock_embeddings = AsyncMock()
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1] * 768])
    store.embeddings = mock_embeddings

    async def _fake_save(fact):
        mem = MagicMock()
        mem.id = "mem_123"
        mem.type = fact.type
        return mem

    store.save = AsyncMock(side_effect=_fake_save)
    return store


def _make_mock_llm(response_content: str):
    llm = AsyncMock()
    response = MagicMock()
    response.content = response_content
    llm.ainvoke = AsyncMock(return_value=response)
    return llm


class TestParseResponse:
    """_parse_response should parse LLM JSON into MemoryFact objects."""

    def test_parses_valid_json_array(self):
        """Given valid JSON array, when parsed, then returns list of MemoryFact."""
        extractor = MemoryExtractor(llm=MagicMock(), store=MagicMock())
        content = json.dumps([{"content": "User likes tea", "type": "user", "confidence": "confirmed"}])

        facts = extractor._parse_response(content)

        assert len(facts) == 1, "Should parse one fact"
        assert facts[0].content == "User likes tea", "Content should match"
        assert facts[0].type == "user", "Type should match"

    def test_returns_empty_for_empty_array(self):
        """Given empty JSON array, when parsed, then returns empty list."""
        extractor = MemoryExtractor(llm=MagicMock(), store=MagicMock())

        facts = extractor._parse_response("[]")

        assert facts == [], "Should return empty list"

    def test_strips_markdown_code_fences(self):
        """Given JSON wrapped in code fences, when parsed, then fences are stripped."""
        extractor = MemoryExtractor(llm=MagicMock(), store=MagicMock())
        content = '```json\n[{"content": "A fact"}]\n```'

        facts = extractor._parse_response(content)

        assert len(facts) == 1, "Should parse through code fences"
        assert facts[0].content == "A fact", "Content should match"

    def test_returns_empty_for_invalid_json(self):
        """Given malformed JSON, when parsed, then returns empty list."""
        extractor = MemoryExtractor(llm=MagicMock(), store=MagicMock())

        facts = extractor._parse_response("not json at all")

        assert facts == [], "Should return empty for invalid JSON"

    def test_returns_empty_for_non_array_json(self):
        """Given a JSON object instead of array, when parsed, then returns empty list."""
        extractor = MemoryExtractor(llm=MagicMock(), store=MagicMock())

        facts = extractor._parse_response('{"content": "not an array"}')

        assert facts == [], "Should return empty for non-array JSON"

    def test_skips_facts_with_missing_content(self):
        """Given a fact dict without content key, when parsed, then it is skipped."""
        extractor = MemoryExtractor(llm=MagicMock(), store=MagicMock())
        content = json.dumps([{"type": "facts"}, {"content": "Valid fact"}])

        facts = extractor._parse_response(content)

        assert len(facts) == 1, "Should skip fact without content"
        assert facts[0].content == "Valid fact", "Should keep valid fact"

    def test_skips_facts_with_invalid_type(self):
        """Given a fact with invalid type, when parsed, then it is skipped."""
        extractor = MemoryExtractor(llm=MagicMock(), store=MagicMock())
        content = json.dumps([{"content": "Bad type", "type": "invalid"}, {"content": "Good fact"}])

        facts = extractor._parse_response(content)

        assert len(facts) == 1, "Should skip fact with invalid type"
        assert facts[0].content == "Good fact", "Should keep valid fact"


class TestBuildPrompt:
    """_build_prompt should assemble system and user messages."""

    def test_includes_user_and_assistant_text(self):
        """Given user and assistant text, when prompt built, then both appear in HumanMessage."""
        extractor = MemoryExtractor(llm=MagicMock(), store=MagicMock())

        messages = extractor._build_prompt("Hello Sol", "Hi there!", "")

        user_content = messages[1].content
        assert "Hello Sol" in user_content, "Should contain user message"
        assert "Hi there!" in user_content, "Should contain assistant response"

    def test_uses_placeholder_for_empty_manifest(self):
        """Given empty manifest, when prompt built, then placeholder text is used."""
        extractor = MemoryExtractor(llm=MagicMock(), store=MagicMock())

        messages = extractor._build_prompt("Hi", "Hello", "")

        assert "(no existing memories)" in messages[1].content, "Should use placeholder"


class TestExtract:
    """extract() should call LLM, parse response, and save facts."""

    async def test_returns_empty_on_llm_failure(self):
        """Given LLM that raises, when extract called, then returns empty list."""
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(side_effect=ConnectionError("refused"))
        store = _make_mock_store()
        extractor = MemoryExtractor(llm=llm, store=store)

        result = await extractor.extract("Hi", "Hello")

        assert result == [], "Should return empty on LLM failure"

    async def test_saves_parsed_facts(self):
        """Given LLM returning valid facts, when extract called, then store.save called for each."""
        response_json = json.dumps(
            [
                {"content": "Fact one", "type": "facts"},
                {"content": "Fact two", "type": "user"},
            ],
        )
        llm = _make_mock_llm(response_json)
        store = _make_mock_store()
        extractor = MemoryExtractor(llm=llm, store=store)

        result = await extractor.extract("Hi", "Hello")

        assert len(result) == 2, "Should return two memory IDs"
        assert store.save.call_count == 2, "Should call save twice"

    async def test_skips_whitespace_only_content(self):
        """Given a fact with whitespace-only content, when saving, then it is skipped."""
        response_json = json.dumps([{"content": "   "}, {"content": "Real fact"}])
        llm = _make_mock_llm(response_json)
        store = _make_mock_store()
        extractor = MemoryExtractor(llm=llm, store=store)

        result = await extractor.extract("Hi", "Hello")

        assert len(result) == 1, "Should skip whitespace-only fact"
        assert store.save.call_count == 1, "Should only save real fact"
