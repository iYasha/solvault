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


class HealthResponse(BaseModel):
    status: str
    version: str
