import structlog
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from sol.config import settings
from sol.database import async_session
from sol.memory.extractor import MemoryExtractor
from sol.memory.store import MemoryStore

log = structlog.get_logger()


async def extract_memories(
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    user_text: str,
    response_text: str,
) -> None:
    """Background task: extract and save memories from a conversation turn."""
    try:
        async with async_session() as db:
            store = MemoryStore(db=db, embeddings=embeddings, config=settings.memory)
            extractor = MemoryExtractor(llm=llm, store=store)
            await extractor.extract(user_text, response_text)
            await db.commit()
    except Exception:
        log.warning("memory.background_extraction_failed", exc_info=True)
