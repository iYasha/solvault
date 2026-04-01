from unittest.mock import patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from sol.config import settings
from sol.core.agent import Agent
from sol.core.llm import get_embeddings, get_llm
from sol.core.prompts import load_system_prompt
from sol.database import Base, VecConnection


class TokenTracker(BaseCallbackHandler):
    """Accumulates token usage across all LLM calls in the eval session."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.call_count: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.call_count += 1
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)


def pytest_configure(config: pytest.Config) -> None:
    config._eval_token_tracker = TokenTracker()


def pytest_terminal_summary(terminalreporter, exitstatus, config: pytest.Config) -> None:
    tracker = getattr(config, "_eval_token_tracker", None)
    if tracker is None or tracker.call_count == 0:
        return
    terminalreporter.section("Eval Token Summary")
    terminalreporter.write_line(f"  LLM calls  : {tracker.call_count}")
    terminalreporter.write_line(f"  Prompt     : {tracker.prompt_tokens:,}")
    terminalreporter.write_line(f"  Completion : {tracker.completion_tokens:,}")
    terminalreporter.write_line(f"  Total      : {tracker.total_tokens:,}")


@pytest.fixture(scope="session")
def token_tracker(request):
    return request.config._eval_token_tracker


@pytest.fixture(scope="session")
def eval_llm(token_tracker):
    """Real ChatOpenAI for the agent, with token tracking."""
    llm = get_llm()
    llm.callbacks = [token_tracker]
    return llm


@pytest.fixture(scope="session")
def eval_judge_llm(token_tracker):
    """Real ChatOpenAI for the judge — temperature=0 for deterministic verdicts."""
    return ChatOpenAI(
        model=settings.llm.model,
        base_url=settings.llm.endpoint,
        api_key=settings.llm.api_key or "not-needed",
        max_tokens=settings.llm.response_token_budget,
        temperature=0.0,
        callbacks=[token_tracker],
    )


@pytest.fixture(scope="session")
def eval_embeddings():
    """Real OpenAIEmbeddings pointed at LM Studio."""
    return get_embeddings()


@pytest.fixture(scope="session")
def eval_agent(eval_llm):
    """Real Agent using the live LLM."""
    return Agent(llm=eval_llm, system_prompt=load_system_prompt())


@pytest.fixture()
async def eval_engine():
    """Fresh in-memory SQLite engine per eval test with all tables and virtual tables."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        echo=False,
        connect_args={"factory": VecConnection},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.exec_driver_sql(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
            "USING fts5(content, memory_id UNINDEXED, tokenize='porter unicode61')",
        )
        dim = settings.memory.embedding.dimensions
        await conn.exec_driver_sql(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec "
            f"USING vec0(memory_id TEXT PRIMARY KEY, embedding float[{dim}])",
        )
    yield engine
    await engine.dispose()


@pytest.fixture()
async def eval_db(eval_engine):
    """AsyncSession backed by the per-test engine."""
    session = AsyncSession(eval_engine, expire_on_commit=False)
    try:
        yield session
    finally:
        await session.close()


@pytest.fixture(autouse=True)
async def _patch_bg_session(eval_engine):
    """Redirect background extraction tasks to use the test engine instead of production DB."""
    eval_session_factory = async_sessionmaker(eval_engine, expire_on_commit=False)
    with patch("sol.memory.tasks.async_session", eval_session_factory):
        yield
