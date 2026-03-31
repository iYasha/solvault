from unittest.mock import patch

import sqlalchemy as sa

from sol.config import IdentityConfig, IdentityMapping
from sol.router.message_router import IncomingMessage, MessageRouter
from sol.session.models import ChannelType, ChatMessage, Session


class TestResolveCanonicalUser:
    """Identity resolution should map platform IDs to canonical user IDs."""

    def _router_with_mappings(self, mappings: list[IdentityMapping]) -> MessageRouter:
        router = MessageRouter()
        identity = IdentityConfig(mappings=mappings)
        self._patch = patch("sol.router.message_router.settings")
        mock_settings = self._patch.start()
        mock_settings.identity = identity
        return router

    def teardown_method(self):
        if hasattr(self, "_patch"):
            self._patch.stop()

    def test_resolves_telegram_identity(self):
        """Given a telegram mapping in config, when resolving, then canonical_id is returned."""
        router = self._router_with_mappings(
            [
                IdentityMapping(canonical_id="user:iyasha", telegram_id="12345"),
            ],
        )
        result = router.resolve_canonical_user(ChannelType.TELEGRAM, "12345")
        assert result == "user:iyasha", "Should resolve to canonical ID from mapping"

    def test_resolves_cli_identity(self):
        """Given a cli mapping in config, when resolving, then canonical_id is returned."""
        router = self._router_with_mappings(
            [
                IdentityMapping(canonical_id="user:iyasha", cli_user="iyasha"),
            ],
        )
        result = router.resolve_canonical_user(ChannelType.CLI, "iyasha")
        assert result == "user:iyasha", "Should resolve to canonical ID from mapping"

    def test_falls_back_to_platform_id(self):
        """Given no mapping, when resolving, then channel:platform_id is returned."""
        router = self._router_with_mappings([])
        result = router.resolve_canonical_user(ChannelType.TELEGRAM, "99999")
        assert result == "telegram:99999", "Should fall back to channel:platform_id format"


class TestRoute:
    """Routing should create sessions, persist messages, and resolve identity."""

    async def test_route_creates_session_and_message(self, db_session):
        """Given a new message, when routed, then session and message are persisted."""
        router = MessageRouter()
        message = IncomingMessage(
            channel=ChannelType.CLI,
            user_id="testuser",
            text="hello sol",
        )

        session, message_id = await router.route(message, db_session)

        assert session.id is not None, "Session should be created"
        assert message_id is not None, "Message should be created"

        session_count = await db_session.execute(sa.select(sa.func.count()).select_from(Session))
        assert session_count.scalar_one() == 1, "One session should exist"

        msg_count = await db_session.execute(sa.select(sa.func.count()).select_from(ChatMessage))
        assert msg_count.scalar_one() == 1, "One message should exist"

    async def test_route_reuses_existing_session(self, db_session):
        """Given two messages from same user, when routed, then same session is used."""
        router = MessageRouter()

        msg1 = IncomingMessage(channel=ChannelType.CLI, user_id="testuser", text="first")
        msg2 = IncomingMessage(channel=ChannelType.CLI, user_id="testuser", text="second")

        session1, _ = await router.route(msg1, db_session)
        session2, _ = await router.route(msg2, db_session)

        assert session1.id == session2.id, "Same user should reuse the same session"

        msg_count = await db_session.execute(sa.select(sa.func.count()).select_from(ChatMessage))
        assert msg_count.scalar_one() == 2, "Two messages should exist in the same session"
