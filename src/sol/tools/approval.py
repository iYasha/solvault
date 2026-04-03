import asyncio
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

import structlog
from fastapi import WebSocket

log = structlog.get_logger()


@dataclass
class ApprovalRequest:
    """Sent to the client when a tool needs user approval."""

    tool_name: str
    display: str
    request_id: str = field(default_factory=lambda: uuid4().hex)


class ApprovalCallback(Protocol):
    """Interface for requesting tool approval from the user."""

    async def request(self, tool_name: str, display: str) -> bool: ...


class DenyAllApprovalCallback:
    """Always denies — used for channels without interactive approval (HTTP, Telegram)."""

    async def request(self, tool_name: str, display: str) -> bool:
        log.info("approval.auto_denied", tool=tool_name)
        return False


class WebSocketApprovalCallback:
    """Sends approval_request over WebSocket, awaits approval_response.

    The WebSocket receive loop must call ``resolve()`` when an
    ``approval_response`` frame arrives.
    """

    def __init__(self, websocket: WebSocket, timeout: float = 30.0) -> None:
        self._ws = websocket
        self._timeout = timeout
        self._pending: dict[str, asyncio.Future[bool]] = {}

    async def request(self, tool_name: str, display: str) -> bool:
        req = ApprovalRequest(tool_name=tool_name, display=display)
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[req.request_id] = future

        await self._ws.send_json(
            {
                "type": "approval_request",
                "request_id": req.request_id,
                "tool": req.tool_name,
                "display": req.display,
            },
        )

        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except TimeoutError:
            log.warning("approval.timeout", tool=tool_name, request_id=req.request_id)
            return False
        finally:
            self._pending.pop(req.request_id, None)

    def resolve(self, request_id: str, approved: bool) -> None:
        """Called by the WS receive loop when approval_response arrives."""
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_result(approved)
