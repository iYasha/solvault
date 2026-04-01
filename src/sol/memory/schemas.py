from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    USER = "user"
    WORK = "work"
    FACTS = "facts"
    CONVERSATIONS = "conversations"
    EVENTS = "events"


class Confidence(StrEnum):
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    UNCERTAIN = "uncertain"


class MemoryFact(BaseModel):
    """A single fact extracted from a conversation turn."""

    content: str = Field(description="The fact text to save")
    type: MemoryType = Field(default=MemoryType.FACTS, description="Category of the memory")
    confidence: Confidence = Field(default=Confidence.INFERRED, description="How certain is this fact")
    tags: list[str] = Field(default_factory=list, description="Relevant categories")
    expires_at: datetime | None = Field(
        default=None,
        description="When this memory expires (ISO format), null if permanent",
    )


class ExtractionResult(BaseModel):
    """Structured response from the memory extraction LLM call."""

    facts: list[MemoryFact] = Field(default_factory=list, description="List of extracted facts")
