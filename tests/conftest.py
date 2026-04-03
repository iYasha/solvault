from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from sol.core.agent import Agent
from sol.database import Base, VecConnection
from sol.gateway.dependencies import get_agent, get_db
from sol.gateway.main import app
from sol.memory import models as _memory_models  # noqa: F401 — register models
from sol.session import models as _models  # noqa: F401 — register models


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
async def test_engine():
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
        await conn.exec_driver_sql(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec "
            "USING vec0(memory_id TEXT PRIMARY KEY, embedding float[768])",
        )
    yield engine
    await engine.dispose()


@pytest.fixture()
async def db_session(test_engine):
    """Per-test DB session with automatic rollback."""
    async with test_engine.connect() as conn:
        await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)

        yield session

        await session.close()
        await conn.rollback()


@pytest.fixture()
def mock_agent():
    """Mock Agent that returns a fixed response."""
    agent = AsyncMock(spec=Agent)
    agent.run = AsyncMock(return_value="Test response from Sol")
    agent.llm = MagicMock()
    return agent


@pytest.fixture()
async def client(db_session, mock_agent):
    """Async HTTP test client with DB session and agent overrides."""

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_agent] = lambda: mock_agent

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c

    app.dependency_overrides.clear()
