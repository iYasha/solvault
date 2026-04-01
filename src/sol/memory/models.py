import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from sol.database import Base


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True, default=lambda: uuid.uuid4().hex)
    type: Mapped[str] = mapped_column(sa.String, nullable=False, default="facts")
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    confidence: Mapped[str] = mapped_column(sa.String, nullable=False, default="inferred")
    source: Mapped[str] = mapped_column(sa.String, nullable=False, default="conversation")
    tags_json: Mapped[str] = mapped_column(sa.String, nullable=False, default="[]")
    embedding: Mapped[bytes | None] = mapped_column(sa.LargeBinary, nullable=True)
    token_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=lambda: datetime.now(UTC))
    expires_at: Mapped[datetime | None] = mapped_column(sa.DateTime, nullable=True, default=None)

    __table_args__ = (sa.Index("idx_memories_type", "type"),)
