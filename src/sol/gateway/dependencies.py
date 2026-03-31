from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from sol.database import async_session


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
