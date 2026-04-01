from collections.abc import AsyncGenerator

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sol.core.errors import AgentError
from sol.core.llm import create_llm
from sol.core.prompts import load_system_prompt
from sol.session.models import ChatMessage, Role

log = structlog.get_logger()


class Agent:
    """LLM-backed conversational agent.

    Converts session history into LangChain messages, calls the LLM,
    and returns the response — optionally as a stream of chunks.
    """

    def __init__(self, llm: ChatOpenAI, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    async def run(self, history: list[ChatMessage]) -> str:
        """Generate a complete response for the given conversation history."""
        messages = self._build_messages(history)
        try:
            response = await self.llm.ainvoke(messages)
        except Exception as exc:
            log.error("agent.run_failed", error=str(exc))
            raise AgentError(str(exc)) from exc
        return str(response.content)

    async def run_stream(self, history: list[ChatMessage]) -> AsyncGenerator[str]:
        """Stream response chunks for the given conversation history."""
        messages = self._build_messages(history)
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

    def _build_messages(self, history: list[ChatMessage]) -> list[BaseMessage]:
        """Convert ORM ChatMessage list to LangChain message format."""
        lc_messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        for msg in history:
            if msg.role == Role.USER:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == Role.ASSISTANT:
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == Role.SYSTEM:
                lc_messages.append(SystemMessage(content=msg.content))
        return lc_messages


def create_agent() -> Agent:
    """Create an Agent with the configured LLM and system prompt."""
    return Agent(llm=create_llm(), system_prompt=load_system_prompt())
