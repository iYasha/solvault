import asyncio
import contextlib
import functools
import json
import os
import select
import sys

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

from sol.config import settings

_PASTE_DEBOUNCE_SEC = 0.05
_RECONNECT_DELAY_SEC = 2
_MAX_RECONNECT_ATTEMPTS = 5


def _read_input(prompt: str) -> str:
    """Read user input, collecting multi-line pastes into a single string."""
    first_line = input(prompt)
    lines = [first_line]

    # Drain any buffered lines that arrived from a paste
    while select.select([sys.stdin], [], [], _PASTE_DEBOUNCE_SEC)[0]:
        line = sys.stdin.readline()
        if not line:
            break
        lines.append(line.rstrip("\n"))

    return "\n".join(lines)


async def _load_history(base_url: str, user_id: str) -> None:
    """Fetch and display existing chat history."""
    url = f"{base_url}/v1/messages/cli/{user_id}/history"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)

    if resp.status_code != 200:
        return

    messages = resp.json().get("messages", [])
    if not messages:
        return

    for msg in messages:
        if msg["role"] == "user":
            print(f"You: {msg['content']}")  # noqa: T201
        elif msg["role"] == "assistant":
            print(f"Sol: {msg['content']}")  # noqa: T201
    print()  # noqa: T201


async def _handle_approval_request(ws: websockets.ClientConnection, frame: dict) -> None:
    """Prompt the user for tool approval and send the response."""
    tool_name = frame.get("tool", "unknown")
    display = frame.get("display", "")
    request_id = frame.get("request_id", "")

    prompt_text = f"\n[Tool: {tool_name}]"
    if display:
        prompt_text += f" {display}"
    prompt_text += "\nAllow? [y/N] "

    sys.stdout.write(prompt_text)
    sys.stdout.flush()

    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, functools.partial(input, ""))
    approved = answer.strip().lower() in ("y", "yes")

    await ws.send(
        json.dumps(
            {
                "type": "approval_response",
                "request_id": request_id,
                "approved": approved,
            },
        ),
    )

    status = "approved" if approved else "denied"
    sys.stdout.write(f"[{status}]\n")
    sys.stdout.flush()


async def _send_and_stream(ws: websockets.ClientConnection, text: str) -> None:
    """Send a message and stream the response."""
    await ws.send(json.dumps({"type": "message", "text": text}))

    sys.stdout.write("Sol: ")
    sys.stdout.flush()

    while True:
        raw = await ws.recv()
        frame = json.loads(raw)

        if frame["type"] == "chunk":
            sys.stdout.write(frame["text"])
            sys.stdout.flush()
        elif frame["type"] == "approval_request":
            await _handle_approval_request(ws, frame)
        elif frame["type"] == "error":
            sys.stdout.write(f"\n[Error: {frame.get('detail', 'Unknown error')}]")
            break
        elif frame["type"] == "done":
            sys.stdout.write("\n\n")
            sys.stdout.flush()
            break


async def _connect(ws_url: str, user_id: str) -> websockets.ClientConnection:
    """Connect to the WebSocket with retry logic."""
    url = f"{ws_url}?channel=cli&user_id={user_id}"
    for attempt in range(_MAX_RECONNECT_ATTEMPTS):
        try:
            return await websockets.connect(url)
        except (OSError, websockets.WebSocketException):
            if attempt == _MAX_RECONNECT_ATTEMPTS - 1:
                raise
            delay = _RECONNECT_DELAY_SEC * (attempt + 1)
            print(f"[Connection failed, retrying in {delay}s...]")  # noqa: T201
            await asyncio.sleep(delay)
    raise ConnectionError("Failed to connect")  # unreachable, satisfies type checker


async def _chat_loop(ws_url: str, user_id: str) -> None:
    base_url = f"http://{settings.gateway.host}:{settings.gateway.port}"
    await _load_history(base_url, user_id)

    ws = await _connect(ws_url, user_id)
    print("Sol — type your message (Ctrl+C to exit)\n")  # noqa: T201

    loop = asyncio.get_event_loop()
    while True:
        try:
            user_input = await loop.run_in_executor(None, functools.partial(_read_input, "You: "))
        except (EOFError, KeyboardInterrupt):
            print()  # noqa: T201
            break

        if not user_input.strip():
            continue

        if user_input.strip() == "/exit":
            break

        try:
            await _send_and_stream(ws, user_input)
        except ConnectionClosed:
            print("\n[Connection lost, reconnecting...]")  # noqa: T201
            try:
                ws = await _connect(ws_url, user_id)
                print("[Reconnected. Resending message...]\n")  # noqa: T201
                await _send_and_stream(ws, user_input)
            except (OSError, websockets.WebSocketException):
                print("[Failed to reconnect. Exiting.]")  # noqa: T201
                break

    await ws.close()


def run_chat() -> None:
    """Start an interactive chat session over WebSocket."""
    ws_url = f"ws://{settings.gateway.host}:{settings.gateway.port}/v1/ws"
    user_id = os.getenv("USER", "anonymous")

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_chat_loop(ws_url, user_id))
