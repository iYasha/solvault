from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from sol.gateway.dependencies import get_db
from sol.gateway.schemas import IncomingMessageRequest, MessageResponse
from sol.router.message_router import IncomingMessage, MessageRouter

router = APIRouter()
message_router = MessageRouter()


@router.post("/messages", response_model=MessageResponse)
async def send_message(
    body: IncomingMessageRequest,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    message = IncomingMessage(
        channel=body.channel,
        user_id=body.user_id,
        text=body.text,
        metadata=body.metadata,
    )
    session, message_id = await message_router.route(message, db)
    return MessageResponse(session_id=session.id, message_id=message_id)
