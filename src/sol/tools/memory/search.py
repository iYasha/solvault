import structlog
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from sol.config import settings
from sol.core.llm import embeddings
from sol.database import async_session
from sol.memory.retriever import MemoryRetriever
from sol.tools.permissions import require_permission

log = structlog.get_logger()


@tool
@require_permission("memory_search", arg="query", display="Search memories: {query}")
async def memory_search(query: str, config: RunnableConfig, top_k: int = 5) -> str:
    """Search your personal memory for relevant facts, user preferences, or stored knowledge.

    Use this for deeper recall beyond what was automatically injected into context.
    """
    log.info("tool.memory_search.execute", query=query, top_k=top_k)
    try:
        async with async_session() as db:
            retriever = MemoryRetriever(db=db, embeddings=embeddings, config=settings.memory)
            results = await retriever.search(query, top_k=top_k)
    except Exception as exc:
        return f"[Error] Memory search failed: {exc}"

    if not results:
        return "No relevant memories found."

    lines = []
    for r in results:
        lines.append(f"[{r.memory_type}] (score: {r.score:.3f}) {r.content}")
    return "\n\n".join(lines)
