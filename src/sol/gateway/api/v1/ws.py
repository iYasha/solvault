import asyncio
import contextlib

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sol.config import settings
from sol.core.agent import Agent
from sol.core.errors import AgentError
from sol.core.llm import embeddings
from sol.database import async_session
from sol.memory.injector import MemoryInjector
from sol.memory.retriever import MemoryRetriever
from sol.memory.tasks import extract_memories
from sol.router.message_router import IncomingMessage, MessageRouter
from sol.session.manager import SessionManager
from sol.session.models import ChannelType, Role
from sol.tools import WebSocketApprovalCallback

router = APIRouter()
log = structlog.get_logger()


class ChatSession:
    """Manages a single WebSocket chat session: routing, memory, streaming, extraction."""

    def __init__(self, websocket: WebSocket, channel: ChannelType, user_id: str) -> None:
        self.websocket = websocket
        self.channel = channel
        self.user_id = user_id
        self.agent: Agent = websocket.app.state.agent
        self.embeddings = embeddings
        self.message_router = MessageRouter()
        self.max_history_tokens = settings.llm.max_context_tokens - settings.llm.response_token_budget
        self.approval_callback = WebSocketApprovalCallback(websocket, timeout=settings.tools.approval_timeout)
        self._background_tasks: set[asyncio.Task[None]] = set()

    async def handle_message(self, text: str) -> None:
        """Process a single message: route, retrieve memories, stream response, save."""
        session_id, history, memory_context = await self._prepare(text)

        full_response = await self._stream_response(session_id, history, memory_context)
        if full_response is None:
            return

        await self._save_response(session_id, full_response)
        self._schedule_memory_extraction(text, full_response)
        await self.websocket.send_json({"type": "done", "session_id": session_id})

    async def _prepare(self, text: str) -> tuple[str, list, str]:
        """Route message, fetch history, and retrieve relevant memories."""
        async with async_session() as db:
            message = IncomingMessage(channel=self.channel, user_id=self.user_id, text=text)
            session, _ = await self.message_router.route(message, db)

            manager = SessionManager(db)
            history = await manager.get_history(
                session_id=session.id,
                max_tokens=self.max_history_tokens,
                model=settings.llm.model,
            )
            memory_context = await self._retrieve_memory_context(db, text)

        return session.id, history, memory_context

    async def _retrieve_memory_context(self, db: object, query: str) -> str:
        """Retrieve relevant memories and format for system prompt injection."""
        try:
            retriever = MemoryRetriever(db=db, embeddings=self.embeddings, config=settings.memory)  # type: ignore[arg-type]
            results = await retriever.search(query)
            return MemoryInjector().build_memory_context(results, settings.memory.injection_max_tokens)
        except Exception:
            log.warning("memory.retrieval_failed", exc_info=True)
            return ""

    async def _stream_response(self, session_id: str, history: list, memory_context: str) -> str | None:
        """Stream agent response chunks over WebSocket. Returns full response or None on error."""
        full_response = ""
        try:
            async for chunk in self.agent.run_stream(
                history,
                memory_context=memory_context,
                approval_callback=self.approval_callback,
            ):
                full_response += chunk
                await self.websocket.send_json({"type": "chunk", "text": chunk})
        except AgentError as exc:
            log.error("ws.agent_error", error=str(exc))
            await self.websocket.send_json({"type": "error", "detail": "Failed to generate response."})
            await self.websocket.send_json({"type": "done", "session_id": session_id})
            return None
        return full_response

    async def _save_response(self, session_id: str, response: str) -> None:
        """Persist the assistant response to the database."""
        async with async_session() as db:
            manager = SessionManager(db)
            await manager.save_message(
                session_id=session_id,
                role=Role.ASSISTANT,
                content=response,
                model=settings.llm.model,
            )
            await db.commit()

    def _schedule_memory_extraction(self, user_text: str, response_text: str) -> None:
        """Fire-and-forget memory extraction."""
        if not settings.memory.extraction_enabled:
            return
        task = asyncio.create_task(extract_memories(self.agent.llm, self.embeddings, user_text, response_text))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)


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

    chat = ChatSession(websocket, channel_type, user_id)

    try:
        await _receive_loop(websocket, chat)
    except WebSocketDisconnect:
        log.info("ws.disconnected", channel=channel_str, user_id=user_id)
    except Exception as e:
        log.error("ws.error", error=str(e))
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "detail": str(e)})


async def _receive_loop(websocket: WebSocket, chat: ChatSession) -> None:
    """WebSocket receive loop that handles messages and approval responses concurrently."""
    message_task: asyncio.Task | None = None

    while True:
        data = await websocket.receive_json()
        frame_type = data.get("type")

        if frame_type == "ping":
            await websocket.send_json({"type": "pong"})
            continue

        if frame_type == "approval_response":
            chat.approval_callback.resolve(
                data.get("request_id", ""),
                data.get("approved", False),
            )
            continue

        if frame_type != "message":
            continue

        text = data.get("text", "")
        if not text:
            continue

        if message_task and not message_task.done():
            await websocket.send_json({"type": "error", "detail": "A message is already being processed."})
            continue

        message_task = asyncio.create_task(chat.handle_message(text))
