import sqlalchemy as sa

from sol.memory.models import Memory


class TestMemoryModel:
    """Memory ORM model should provide sensible defaults."""

    async def test_default_id_is_generated(self, db_session):
        """Given a Memory without explicit id, when flushed, then id is a non-empty hex string."""
        memory = Memory(content="Test content")
        db_session.add(memory)
        await db_session.flush()

        assert memory.id is not None, "ID should be generated"
        assert len(memory.id) == 32, "ID should be a 32-char hex string"

    async def test_default_type_is_facts(self, db_session):
        """Given a Memory without type, when saved, then type defaults to facts."""
        memory = Memory(content="Test content")
        db_session.add(memory)
        await db_session.flush()

        assert memory.type == "facts", "Default type should be facts"

    async def test_default_confidence_is_inferred(self, db_session):
        """Given a Memory without confidence, when saved, then confidence defaults to inferred."""
        memory = Memory(content="Test content")
        db_session.add(memory)
        await db_session.flush()

        assert memory.confidence == "inferred", "Default confidence should be inferred"

    async def test_default_source_is_conversation(self, db_session):
        """Given a Memory without source, when saved, then source defaults to conversation."""
        memory = Memory(content="Test content")
        db_session.add(memory)
        await db_session.flush()

        assert memory.source == "conversation", "Default source should be conversation"

    async def test_created_at_is_set(self, db_session):
        """Given a Memory without created_at, when saved, then created_at is populated."""
        memory = Memory(content="Test content")
        db_session.add(memory)
        await db_session.flush()

        assert memory.created_at is not None, "created_at should be auto-set"

    async def test_persists_to_db(self, db_session):
        """Given a saved Memory, when queried, then it exists in DB."""
        memory = Memory(content="Persistence test")
        db_session.add(memory)
        await db_session.flush()

        result = await db_session.execute(sa.select(sa.func.count()).select_from(Memory).where(Memory.id == memory.id))
        assert result.scalar_one() == 1, "Memory should exist in DB"
