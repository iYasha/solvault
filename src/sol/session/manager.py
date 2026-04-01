from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from sol.session.models import ChannelType, ChatMessage, Role, Session
from sol.session.token_window import apply_token_window, count_tokens


class SessionManager:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_or_create_session(self, channel: ChannelType, user_id: str) -> Session:
        result = await self.db.execute(
            sa.select(Session).where(
                Session.channel == channel,
                Session.user_id == user_id,
            ),
        )
        session = result.scalar_one_or_none()

        if session is None:
            session = Session(channel=channel, user_id=user_id)
            self.db.add(session)
            await self.db.flush()

        return session

    async def save_message(
        self,
        session_id: str,
        role: Role,
        content: str,
        model: str = "gpt-4",
    ) -> ChatMessage:
        token_count = count_tokens(content, model)
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            token_count=token_count,
        )
        self.db.add(message)

        await self.db.execute(
            sa.update(Session).where(Session.id == session_id).values(updated_at=datetime.now(UTC)),
        )

        await self.db.flush()
        return message

    async def get_history(
        self,
        session_id: str,
        max_tokens: int | None = None,
        model: str = "gpt-4",
    ) -> list[ChatMessage]:
        result = await self.db.execute(
            sa.select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.asc()),
        )
        messages = list(result.scalars().all())
        if max_tokens is not None:
            return apply_token_window(messages, max_tokens, model)
        return messages
