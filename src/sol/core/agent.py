from collections.abc import AsyncGenerator
from functools import cached_property

import structlog
from langchain.agents import create_agent as create_langchain_agent
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sol.config import settings
from sol.core.errors import AgentError
from sol.core.llm import get_llm
from sol.core.prompts import load_system_prompt
from sol.session.models import ChatMessage, Role
from sol.tools import ALL_TOOLS
from sol.tools.approval import ApprovalCallback
from sol.tools.permissions import PermissionGate

log = structlog.get_logger()


class Agent:
    """LLM-backed conversational agent with tool support.

    The LangGraph agent graph is built lazily on first use via ``cached_property``.
    Per-request permission context flows via RunnableConfig at invocation time.
    """

    def __init__(self, llm: ChatOpenAI, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    @cached_property
    def agent(self) -> object:
        """LangGraph react agent, built lazily on first access."""
        return create_langchain_agent(self.llm, ALL_TOOLS)

    async def run(
        self,
        history: list[ChatMessage],
        memory_context: str = "",
        approval_callback: ApprovalCallback | None = None,
    ) -> str:
        """Generate a complete response, using tools if available."""
        messages = self._build_messages(history, memory_context)

        if approval_callback and ALL_TOOLS and settings.tools.enabled:
            return await self._run_with_tools(messages, approval_callback)

        try:
            response = await self.llm.ainvoke(messages)
        except Exception as exc:
            log.error("agent.run_failed", error=str(exc))
            raise AgentError(str(exc)) from exc
        return str(response.content)

    async def run_stream(
        self,
        history: list[ChatMessage],
        memory_context: str = "",
        approval_callback: ApprovalCallback | None = None,
    ) -> AsyncGenerator[str]:
        """Stream response chunks, using tools if available."""
        messages = self._build_messages(history, memory_context)

        if approval_callback and ALL_TOOLS and settings.tools.enabled:
            async for chunk in self._stream_with_tools(messages, approval_callback):
                yield chunk
            return

        try:
            async for chunk in self.llm.astream(messages):
                text = str(chunk.content) if chunk.content else ""
                if text:
                    yield text
        except AgentError:
            raise
        except Exception as exc:
            log.error("agent.stream_failed", error=str(exc))
            raise AgentError(str(exc)) from exc

    async def _run_with_tools(self, messages: list[BaseMessage], approval_callback: ApprovalCallback) -> str:
        """Run a full tool-calling loop via the LangGraph agent."""
        config = _build_config(approval_callback)
        try:
            result = await self.agent.ainvoke({"messages": messages}, config=config)
        except Exception as exc:
            log.error("agent.run_with_tools_failed", error=str(exc))
            raise AgentError(str(exc)) from exc

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                return str(msg.content)
        return ""

    async def _stream_with_tools(
        self,
        messages: list[BaseMessage],
        approval_callback: ApprovalCallback,
    ) -> AsyncGenerator[str]:
        """Stream through a tool-calling loop, yielding only final text chunks."""
        config = _build_config(approval_callback)
        try:
            async for event in self.agent.astream_events(
                {"messages": messages},
                config=config,
                version="v2",
            ):
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if isinstance(chunk, AIMessageChunk) and chunk.content and not chunk.tool_calls:
                        yield str(chunk.content)
        except AgentError:
            raise
        except Exception as exc:
            log.error("agent.stream_with_tools_failed", error=str(exc))
            raise AgentError(str(exc)) from exc

    def _build_messages(self, history: list[ChatMessage], memory_context: str = "") -> list[BaseMessage]:
        """Convert ORM ChatMessage list to LangChain message format."""
        prompt = self.system_prompt + memory_context if memory_context else self.system_prompt
        lc_messages: list[BaseMessage] = [SystemMessage(content=prompt)]
        for msg in history:
            if msg.role == Role.USER:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == Role.ASSISTANT:
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == Role.SYSTEM:
                lc_messages.append(SystemMessage(content=msg.content))
        return lc_messages


def _build_config(approval_callback: ApprovalCallback) -> dict:
    """Build RunnableConfig with permission context."""
    return {
        "configurable": {
            "gate": PermissionGate(settings.tools),
            "approval_callback": approval_callback,
        },
    }


def create_agent() -> Agent:
    """Create an Agent with the configured LLM and system prompt."""
    return Agent(llm=get_llm(), system_prompt=load_system_prompt())
