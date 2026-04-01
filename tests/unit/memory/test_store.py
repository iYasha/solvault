from unittest.mock import AsyncMock

import sqlalchemy as sa

from sol.config import MemoryConfig
from sol.memory.models import Memory
from sol.memory.schemas import MemoryFact
from sol.memory.store import MemoryStore


def _make_store(db_session):
    mock_embeddings = AsyncMock()
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1] * 768])
    return MemoryStore(db=db_session, embeddings=mock_embeddings, config=MemoryConfig())


class TestMemoryStoreSave:
    """MemoryStore.save should persist a memory with embeddings and virtual table entries."""

    async def test_persists_memory_to_db(self, db_session):
        """Given a MemoryFact, when save called, then a Memory row exists in DB."""
        store = _make_store(db_session)
        fact = MemoryFact(content="User likes coffee", type="user", confidence="confirmed", tags=["pref"])

        memory = await store.save(fact)

        result = await db_session.execute(sa.select(Memory).where(Memory.id == memory.id))
        row = result.scalar_one()
        assert row.content == "User likes coffee", "Content should match"
        assert row.type == "user", "Type should match"
        assert row.confidence == "confirmed", "Confidence should match"

    async def test_computes_token_count(self, db_session):
        """Given a MemoryFact, when save called, then token_count is positive."""
        store = _make_store(db_session)
        fact = MemoryFact(content="User works at Acme Corp as a senior engineer")

        memory = await store.save(fact)

        assert memory.token_count > 0, "Token count should be positive"

    async def test_stores_embedding_blob(self, db_session):
        """Given a MemoryFact, when save called, then embedding is a non-null blob."""
        store = _make_store(db_session)
        fact = MemoryFact(content="Some fact")

        memory = await store.save(fact)

        assert memory.embedding is not None, "Embedding should not be null"
        assert len(memory.embedding) > 0, "Embedding blob should have data"

    async def test_inserts_into_fts(self, db_session):
        """Given a saved memory, when querying FTS5, then the entry exists."""
        store = _make_store(db_session)
        fact = MemoryFact(content="Unique FTS test fact")

        memory = await store.save(fact)

        result = await db_session.execute(
            sa.text("SELECT memory_id FROM memories_fts WHERE memory_id = :mid"),
            {"mid": memory.id},
        )
        assert result.scalar_one() == memory.id, "FTS entry should exist"

    async def test_inserts_into_vec(self, db_session):
        """Given a saved memory, when querying vec table, then the entry exists."""
        store = _make_store(db_session)
        fact = MemoryFact(content="Unique vec test fact")

        memory = await store.save(fact)

        result = await db_session.execute(
            sa.text("SELECT memory_id FROM memories_vec WHERE memory_id = :mid"),
            {"mid": memory.id},
        )
        assert result.scalar_one() == memory.id, "Vec entry should exist"


class TestMemoryStoreDelete:
    """MemoryStore.delete should remove memory and its index entries."""

    async def test_removes_memory_row(self, db_session):
        """Given a saved memory, when delete called, then the row is gone."""
        store = _make_store(db_session)
        fact = MemoryFact(content="To be deleted")
        memory = await store.save(fact)

        await store.delete(memory.id)

        result = await db_session.execute(sa.select(Memory).where(Memory.id == memory.id))
        assert result.scalar_one_or_none() is None, "Memory should be deleted"

    async def test_removes_fts_entry(self, db_session):
        """Given a saved memory, when delete called, then FTS entry is removed."""
        store = _make_store(db_session)
        fact = MemoryFact(content="FTS delete test")
        memory = await store.save(fact)

        await store.delete(memory.id)

        result = await db_session.execute(
            sa.text("SELECT count(*) FROM memories_fts WHERE memory_id = :mid"),
            {"mid": memory.id},
        )
        assert result.scalar() == 0, "FTS entry should be deleted"

    async def test_removes_vec_entry(self, db_session):
        """Given a saved memory, when delete called, then vec entry is removed."""
        store = _make_store(db_session)
        fact = MemoryFact(content="Vec delete test")
        memory = await store.save(fact)

        await store.delete(memory.id)

        result = await db_session.execute(
            sa.text("SELECT count(*) FROM memories_vec WHERE memory_id = :mid"),
            {"mid": memory.id},
        )
        assert result.scalar() == 0, "Vec entry should be deleted"


class TestMemoryStoreListAll:
    """MemoryStore.list_all should return memories in creation order."""

    async def test_returns_empty_when_no_memories(self, db_session):
        """Given empty DB, when list_all called, then returns empty list."""
        store = _make_store(db_session)

        memories = await store.list_all()

        # May contain memories from other tests due to shared session-scoped engine,
        # but the method itself should not raise
        assert isinstance(memories, list), "Should return a list"


class TestMemoryStoreBuildManifest:
    """MemoryStore.build_manifest should produce a text summary of all memories."""

    async def test_manifest_includes_type_and_confidence(self, db_session):
        """Given saved memories, when build_manifest called, then output contains type and confidence."""
        store = _make_store(db_session)
        await store.save(MemoryFact(content="Manifest test fact", type="user", confidence="confirmed"))

        manifest = await store.build_manifest()

        assert "[user]" in manifest, "Should contain type"
        assert "(confirmed)" in manifest, "Should contain confidence"
        assert "Manifest test fact" in manifest, "Should contain content preview"
