from unittest.mock import AsyncMock

from sol.config import MemoryConfig
from sol.memory.retriever import MemoryRetriever
from sol.memory.schemas import MemoryFact
from sol.memory.store import MemoryStore


def _make_deps(db_session):
    mock_embeddings = AsyncMock()
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1] * 768])
    mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)
    config = MemoryConfig()
    return mock_embeddings, config


class TestMemoryRetrieverSearch:
    """MemoryRetriever.search should find relevant memories via hybrid search."""

    async def test_returns_empty_when_no_memories(self, db_session):
        """Given empty DB, when search called, then returns empty list."""
        embeddings, config = _make_deps(db_session)
        retriever = MemoryRetriever(db=db_session, embeddings=embeddings, config=config)

        results = await retriever.search("hello")

        assert results == [], "Should return empty for empty DB"

    async def test_finds_saved_memory_by_keyword(self, db_session):
        """Given a saved memory, when search matches keyword, then memory is returned."""
        embeddings, config = _make_deps(db_session)
        store = MemoryStore(db=db_session, embeddings=embeddings, config=config)
        await store.save(MemoryFact(content="User works at Acme Corporation as a developer"))

        retriever = MemoryRetriever(db=db_session, embeddings=embeddings, config=config)
        results = await retriever.search("Acme")

        assert len(results) >= 1, "Should find at least one result"
        assert any("Acme" in r.content for r in results), "Result should contain the keyword"

    async def test_finds_saved_memory_by_vector(self, db_session):
        """Given a saved memory with matching embedding, when search called, then memory is returned."""
        embeddings, config = _make_deps(db_session)
        store = MemoryStore(db=db_session, embeddings=embeddings, config=config)
        await store.save(MemoryFact(content="User enjoys hiking in the mountains"))

        retriever = MemoryRetriever(db=db_session, embeddings=embeddings, config=config)
        results = await retriever.search("outdoor activities")

        # With identical mock embeddings, vector search should match
        assert len(results) >= 1, "Should find at least one result via vector search"

    async def test_respects_top_k(self, db_session):
        """Given many memories, when search with top_k=2, then at most 2 results returned."""
        embeddings, config = _make_deps(db_session)
        store = MemoryStore(db=db_session, embeddings=embeddings, config=config)
        for i in range(5):
            await store.save(MemoryFact(content=f"Topk test memory number {i}"))

        retriever = MemoryRetriever(db=db_session, embeddings=embeddings, config=config)
        results = await retriever.search("topk test memory", top_k=2)

        assert len(results) <= 2, "Should return at most top_k results"

    async def test_result_has_content_and_type(self, db_session):
        """Given a found memory, when inspecting result, then content and type are present."""
        embeddings, config = _make_deps(db_session)
        store = MemoryStore(db=db_session, embeddings=embeddings, config=config)
        await store.save(MemoryFact(content="User prefers dark mode", type="user"))

        retriever = MemoryRetriever(db=db_session, embeddings=embeddings, config=config)
        results = await retriever.search("dark mode")

        matching = [r for r in results if "dark mode" in r.content]
        assert len(matching) >= 1, "Should find the memory"
        assert matching[0].memory_type == "user", "Memory type should be preserved"
        assert matching[0].score > 0, "Score should be positive"
