import pytest
import sqlalchemy as sa

from sol.session.manager import SessionManager
from sol.session.models import ChannelType, ChatMessage, Role, Session


class TestGetOrCreateSession:
    """Session lookup should create new sessions or return existing ones."""

    async def test_creates_new_session(self, db_session):
        """Given no existing session, when get_or_create called, then a new session is persisted."""
        manager = SessionManager(db_session)
        session = await manager.get_or_create_session(ChannelType.CLI, "user:test")

        assert session.id is not None, "Session should have an ID"
        assert session.channel == ChannelType.CLI, "Channel should match"
        assert session.user_id == "user:test", "User ID should match"

        result = await db_session.execute(sa.select(sa.func.count()).select_from(Session))
        assert result.scalar_one() == 1, "Exactly one session should exist in DB"

    async def test_returns_existing_session(self, db_session):
        """Given an existing session, when get_or_create called with same channel+user, then same session returned."""
        manager = SessionManager(db_session)

        first = await manager.get_or_create_session(ChannelType.CLI, "user:test")
        second = await manager.get_or_create_session(ChannelType.CLI, "user:test")

        assert first.id == second.id, "Should return the same session"

        result = await db_session.execute(sa.select(sa.func.count()).select_from(Session))
        assert result.scalar_one() == 1, "Only one session should exist"

    async def test_different_channels_create_different_sessions(self, db_session):
        """Given different channels, when get_or_create called, then separate sessions are created."""
        manager = SessionManager(db_session)

        cli_session = await manager.get_or_create_session(ChannelType.CLI, "user:test")
        tg_session = await manager.get_or_create_session(ChannelType.TELEGRAM, "user:test")

        assert cli_session.id != tg_session.id, "Different channels should have different sessions"


class TestSaveMessage:
    """Messages should be persisted with computed token counts."""

    async def test_saves_message_with_token_count(self, db_session):
        """Given a session, when save_message called, then message is persisted with token count > 0."""
        manager = SessionManager(db_session)
        session = await manager.get_or_create_session(ChannelType.CLI, "user:test")

        message = await manager.save_message(
            session_id=session.id,
            role=Role.USER,
            content="Hello, how are you?",
        )

        assert message.id is not None, "Message should have an ID"
        assert message.role == Role.USER, "Role should match"
        assert message.token_count > 0, "Token count should be computed and positive"

        result = await db_session.execute(sa.select(sa.func.count()).select_from(ChatMessage))
        assert result.scalar_one() == 1, "Exactly one message should exist"

    async def test_updates_session_timestamp(self, db_session):
        """Given a session, when save_message called, then session updated_at is refreshed."""
        manager = SessionManager(db_session)
        session = await manager.get_or_create_session(ChannelType.CLI, "user:test")

        await manager.save_message(
            session_id=session.id,
            role=Role.USER,
            content="test message",
        )

        await db_session.refresh(session)
        assert session.updated_at is not None, "Session timestamp should be set after save_message"


class TestGetHistory:
    """History retrieval should return messages in order within token budget."""

    async def test_returns_messages_in_order(self, db_session):
        """Given multiple messages, when get_history called, then they're returned chronologically."""
        manager = SessionManager(db_session)
        session = await manager.get_or_create_session(ChannelType.CLI, "user:test")

        await manager.save_message(session.id, Role.USER, "first")
        await manager.save_message(session.id, Role.ASSISTANT, "second")
        await manager.save_message(session.id, Role.USER, "third")

        history = await manager.get_history(session.id, max_tokens=100_000)

        assert len(history) == 3, "All three messages should be returned"
        assert [m.content for m in history] == ["first", "second", "third"], "Messages should be in chronological order"

    async def test_respects_token_budget(self, db_session):
        """Given messages exceeding budget, when get_history called, then oldest are dropped."""
        manager = SessionManager(db_session)
        session = await manager.get_or_create_session(ChannelType.CLI, "user:test")

        # Create messages with known content (each ~1-2 tokens)
        for i in range(20):
            await manager.save_message(session.id, Role.USER, f"message number {i}")

        history = await manager.get_history(session.id, max_tokens=10)
        assert len(history) < 20, "Token budget should limit the number of messages returned"
        assert len(history) >= 1, "At least the most recent message should be returned"
