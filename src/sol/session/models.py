import uuid
from datetime import UTC, datetime
from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sol.database import Base


class ChannelType(StrEnum):
    CLI = "cli"
    TELEGRAM = "telegram"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True, default=lambda: uuid.uuid4().hex)
    channel: Mapped[str] = mapped_column(sa.String, nullable=False)
    user_id: Mapped[str] = mapped_column(sa.String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    metadata_json: Mapped[str] = mapped_column(sa.String, default="{}")

    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (sa.Index("idx_sessions_channel_user", "channel", "user_id"),)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True, default=lambda: uuid.uuid4().hex)
    session_id: Mapped[str] = mapped_column(sa.ForeignKey("sessions.id"), nullable=False)
    role: Mapped[str] = mapped_column(sa.String, nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(sa.DateTime, default=lambda: datetime.now(UTC))
    token_count: Mapped[int] = mapped_column(sa.Integer, default=0)

    session: Mapped["Session"] = relationship(back_populates="messages")

    __table_args__ = (sa.Index("idx_messages_session_id", "session_id"),)
