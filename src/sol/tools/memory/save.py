import structlog
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from sol.config import settings
from sol.core.llm import embeddings
from sol.database import async_session
from sol.memory.schemas import Confidence, MemoryFact, MemoryType
from sol.memory.store import MemoryStore
from sol.tools.permissions import require_permission

log = structlog.get_logger()


@tool
@require_permission("memory_save", arg="content", display="Save memory: {content}")
async def memory_save(
    content: str,
    config: RunnableConfig,
    memory_type: MemoryType = MemoryType.FACTS,
    confidence: Confidence = Confidence.INFERRED,
    tags: list[str] | None = None,
) -> str:
    """Save a new fact or piece of information to personal memory.

    Use this to explicitly store important information the user shares,
    preferences, or facts you want to recall later.
    """
    log.info("tool.memory_save.execute", type=memory_type, confidence=confidence)
    try:
        fact = MemoryFact(content=content, type=memory_type, confidence=confidence, tags=tags or [])
        async with async_session() as db:
            store = MemoryStore(db=db, embeddings=embeddings, config=settings.memory)
            memory = await store.save(fact)
            await db.commit()
    except Exception as exc:
        return f"[Error] Failed to save memory: {exc}"

    return f"Memory saved successfully (id: {memory.id}, type: {memory_type})"
