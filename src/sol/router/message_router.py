from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from sol.config import settings
from sol.session.manager import SessionManager
from sol.session.models import ChannelType, Role, Session


@dataclass
class IncomingMessage:
    channel: ChannelType
    user_id: str
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)


class MessageRouter:
    def resolve_canonical_user(self, channel: ChannelType, platform_user_id: str) -> str:
        for mapping in settings.identity.mappings:
            if channel == ChannelType.TELEGRAM and mapping.telegram_id == platform_user_id:
                return mapping.canonical_id
            if channel == ChannelType.CLI and mapping.cli_user == platform_user_id:
                return mapping.canonical_id
        return f"{channel}:{platform_user_id}"

    async def route(self, message: IncomingMessage, db: AsyncSession) -> tuple[Session, str]:
        """Route a message to the correct session. Returns (session, message_id)."""
        canonical_user = self.resolve_canonical_user(message.channel, message.user_id)
        manager = SessionManager(db)

        session = await manager.get_or_create_session(message.channel, canonical_user)
        chat_message = await manager.save_message(
            session_id=session.id,
            role=Role.USER,
            content=message.text,
        )
        await db.commit()

        return session, chat_message.id
