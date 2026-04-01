import contextlib

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sol.config import settings
from sol.core.errors import AgentError
from sol.database import async_session
from sol.router.message_router import IncomingMessage, MessageRouter
from sol.session.manager import SessionManager
from sol.session.models import ChannelType, Role

router = APIRouter()
log = structlog.get_logger()
message_router = MessageRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    channel_str = websocket.query_params.get("channel", "cli")
    user_id = websocket.query_params.get("user_id", "anonymous")

    try:
        channel_type = ChannelType(channel_str)
    except ValueError:
        await websocket.close(code=1008, reason=f"Unknown channel: {channel_str}")
        return

    await websocket.accept()
    log.info("ws.connected", channel=channel_str, user_id=user_id)

    agent = websocket.app.state.agent
    max_history_tokens = settings.llm.max_context_tokens - settings.llm.response_token_budget

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if data.get("type") != "message":
                continue

            text = data.get("text", "")
            if not text:
                continue

            # Route message and fetch history
            async with async_session() as db:
                message = IncomingMessage(
                    channel=channel_type,
                    user_id=user_id,
                    text=text,
                )
                session, _ = await message_router.route(message, db)
                session_id = session.id

                manager = SessionManager(db)
                history = await manager.get_history(
                    session_id=session_id,
                    max_tokens=max_history_tokens,
                    model=settings.llm.model,
                )

            # Stream agent response (no DB session held)
            full_response = ""
            try:
                async for chunk in agent.run_stream(history):
                    full_response += chunk
                    await websocket.send_json({"type": "chunk", "text": chunk})
            except AgentError as exc:
                log.error("ws.agent_error", error=str(exc))
                await websocket.send_json({"type": "error", "detail": "Failed to generate response."})
                await websocket.send_json({"type": "done", "session_id": session_id})
                continue

            # Save complete response
            async with async_session() as db:
                manager = SessionManager(db)
                await manager.save_message(
                    session_id=session_id,
                    role=Role.ASSISTANT,
                    content=full_response,
                    model=settings.llm.model,
                )
                await db.commit()

            await websocket.send_json({"type": "done", "session_id": session_id})

    except WebSocketDisconnect:
        log.info("ws.disconnected", channel=channel_str, user_id=user_id)
    except Exception as e:
        log.error("ws.error", error=str(e))
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "detail": str(e)})
