from collections.abc import AsyncGenerator

from fastapi import Request
from langchain_openai import OpenAIEmbeddings
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


def get_embeddings(request: Request) -> OpenAIEmbeddings:
    """Retrieve the OpenAIEmbeddings instance from application state."""
    return request.app.state.embeddings
