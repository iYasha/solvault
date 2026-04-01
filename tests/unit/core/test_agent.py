from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sol.core.agent import Agent
from sol.core.errors import AgentError
from sol.session.models import Role


def _make_message(role: str, content: str) -> MagicMock:
    """Create a mock ChatMessage with the given role and content."""
    msg = MagicMock()
    msg.role = role
    msg.content = content
    return msg


def _make_llm(response_content: str = "Hello!") -> AsyncMock:
    """Create a mock ChatOpenAI that returns a fixed response."""
    llm = AsyncMock()
    response = MagicMock()
    response.content = response_content
    llm.ainvoke = AsyncMock(return_value=response)
    return llm


class TestAgentRun:
    """Agent.run should invoke the LLM and return the response content."""

    async def test_returns_llm_response(self):
        """Given a valid history, when run called, then LLM response content is returned."""
        llm = _make_llm("Hello from Sol!")
        agent = Agent(llm=llm, system_prompt="You are Sol.")
        history = [_make_message(Role.USER, "Hi")]

        result = await agent.run(history)

        assert result == "Hello from Sol!", "Should return the LLM response content"

    async def test_prepends_system_prompt(self):
        """Given a system prompt, when run called, then first message is SystemMessage."""
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="Be helpful.")
        history = [_make_message(Role.USER, "Hi")]

        await agent.run(history)

        messages = llm.ainvoke.call_args[0][0]
        assert isinstance(messages[0], SystemMessage), "First message should be SystemMessage"
        assert messages[0].content == "Be helpful.", "System prompt should match"

    async def test_converts_history_roles(self):
        """Given history with user/assistant/system messages, when run called, then roles map correctly."""
        llm = _make_llm()
        agent = Agent(llm=llm, system_prompt="System.")
        history = [
            _make_message(Role.USER, "Hello"),
            _make_message(Role.ASSISTANT, "Hi there"),
            _make_message(Role.SYSTEM, "Context"),
            _make_message(Role.USER, "Follow up"),
        ]

        await agent.run(history)

        messages = llm.ainvoke.call_args[0][0]
        assert isinstance(messages[0], SystemMessage), "Index 0 should be system prompt"
        assert isinstance(messages[1], HumanMessage), "Index 1 should be HumanMessage"
        assert isinstance(messages[2], AIMessage), "Index 2 should be AIMessage"
        assert isinstance(messages[3], SystemMessage), "Index 3 should be SystemMessage"
        assert isinstance(messages[4], HumanMessage), "Index 4 should be HumanMessage"

    async def test_raises_agent_error_on_failure(self):
        """Given LLM that raises, when run called, then AgentError is raised."""
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(side_effect=ConnectionError("refused"))
        agent = Agent(llm=llm, system_prompt="System.")

        with pytest.raises(AgentError):
            await agent.run([_make_message(Role.USER, "Hi")])


class TestAgentRunStream:
    """Agent.run_stream should yield LLM response chunks."""

    async def test_yields_chunks(self):
        """Given a streaming LLM, when run_stream called, then chunks are yielded."""
        chunk1 = MagicMock()
        chunk1.content = "Hello "
        chunk2 = MagicMock()
        chunk2.content = "world!"

        llm = AsyncMock()

        async def _fake_stream(messages):
            yield chunk1
            yield chunk2

        llm.astream = _fake_stream
        agent = Agent(llm=llm, system_prompt="System.")

        chunks = [c async for c in agent.run_stream([_make_message(Role.USER, "Hi")])]

        assert chunks == ["Hello ", "world!"], "Should yield all non-empty chunks"

    async def test_skips_empty_chunks(self):
        """Given chunks with empty content, when run_stream called, then empty chunks are skipped."""
        chunk1 = MagicMock()
        chunk1.content = "Hello"
        chunk2 = MagicMock()
        chunk2.content = ""
        chunk3 = MagicMock()
        chunk3.content = None

        llm = AsyncMock()

        async def _fake_stream(messages):
            yield chunk1
            yield chunk2
            yield chunk3

        llm.astream = _fake_stream
        agent = Agent(llm=llm, system_prompt="System.")

        chunks = [c async for c in agent.run_stream([_make_message(Role.USER, "Hi")])]

        assert chunks == ["Hello"], "Should skip empty and None chunks"

    async def test_raises_agent_error_on_failure(self):
        """Given LLM stream that raises, when run_stream called, then AgentError is raised."""
        llm = AsyncMock()

        async def _failing_stream(messages):
            yield MagicMock(content="partial")
            raise ConnectionError("connection lost")

        llm.astream = _failing_stream
        agent = Agent(llm=llm, system_prompt="System.")

        with pytest.raises(AgentError):
            chunks = []
            async for c in agent.run_stream([_make_message(Role.USER, "Hi")]):
                chunks.append(c)
