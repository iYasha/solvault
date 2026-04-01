import asyncio
import contextlib
import functools
import json
import os
import sys

import httpx
import websockets

from sol.config import settings


def _read_input(prompt: str) -> str:
    return input(prompt)


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


async def _chat_loop(ws_url: str, user_id: str) -> None:
    base_url = f"http://{settings.server.host}:{settings.server.port}"
    await _load_history(base_url, user_id)

    async with websockets.connect(f"{ws_url}?channel=cli&user_id={user_id}") as ws:
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

            await ws.send(json.dumps({"type": "message", "text": user_input}))

            sys.stdout.write("Sol: ")
            sys.stdout.flush()

            while True:
                raw = await ws.recv()
                frame = json.loads(raw)

                if frame["type"] == "chunk":
                    sys.stdout.write(frame["text"])
                    sys.stdout.flush()
                elif frame["type"] == "error":
                    sys.stdout.write(f"\n[Error: {frame.get('detail', 'Unknown error')}]")
                    break
                elif frame["type"] == "done":
                    sys.stdout.write("\n\n")
                    sys.stdout.flush()
                    break


def run_chat() -> None:
    """Start an interactive chat session over WebSocket."""
    ws_url = f"ws://{settings.server.host}:{settings.server.port}/v1/ws"
    user_id = os.getenv("USER", "anonymous")

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_chat_loop(ws_url, user_id))
