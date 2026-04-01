from pydantic import BaseModel, Field

from sol.session.models import ChannelType


class IncomingMessageRequest(BaseModel):
    channel: ChannelType
    user_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class MessageResponse(BaseModel):
    session_id: str
    message_id: str
    response_text: str = ""


class ChatMessageOut(BaseModel):
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessageOut]


class HealthResponse(BaseModel):
    status: str
    version: str
