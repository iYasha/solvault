from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from sol.config import settings
from sol.core.agent import Agent
from sol.core.errors import AgentError
from sol.gateway.dependencies import get_agent, get_db
from sol.gateway.schemas import ChatMessageOut, IncomingMessageRequest, MessageResponse, SessionHistoryResponse
from sol.router.message_router import IncomingMessage, MessageRouter
from sol.session.manager import SessionManager
from sol.session.models import ChannelType, Role

router = APIRouter()
message_router = MessageRouter()


@router.post("/messages", response_model=MessageResponse)
async def send_message(
    body: IncomingMessageRequest,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(get_agent),
) -> MessageResponse:
    message = IncomingMessage(
        channel=body.channel,
        user_id=body.user_id,
        text=body.text,
        metadata=body.metadata,
    )
    session, message_id = await message_router.route(message, db)

    manager = SessionManager(db)
    max_history_tokens = settings.llm.max_context_tokens - settings.llm.response_token_budget
    history = await manager.get_history(
        session_id=session.id,
        max_tokens=max_history_tokens,
        model=settings.llm.model,
    )

    try:
        response_text = await agent.run(history)
    except AgentError as exc:
        raise HTTPException(status_code=502, detail="Agent failed to generate response.") from exc

    await manager.save_message(
        session_id=session.id,
        role=Role.ASSISTANT,
        content=response_text,
        model=settings.llm.model,
    )
    await db.commit()

    return MessageResponse(
        session_id=session.id,
        message_id=message_id,
        response_text=response_text,
    )


@router.get("/messages/{channel}/{user_id}/history", response_model=SessionHistoryResponse)
async def get_history(
    channel: ChannelType,
    user_id: str,
    db: AsyncSession = Depends(get_db),
) -> SessionHistoryResponse:
    """Return full chat history for a channel+user session."""
    canonical = message_router.resolve_canonical_user(channel, user_id)
    manager = SessionManager(db)
    session = await manager.get_or_create_session(channel, canonical)
    messages = await manager.get_history(session_id=session.id)
    return SessionHistoryResponse(
        session_id=session.id,
        messages=[ChatMessageOut(role=msg.role, content=msg.content) for msg in messages],
    )
