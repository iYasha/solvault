from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from sol.core.agent import Agent
from sol.database import async_session


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_agent(request: Request) -> Agent:
    """Retrieve the Agent instance from application state."""
    return request.app.state.agent
