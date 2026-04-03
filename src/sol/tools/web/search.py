import asyncio

import structlog
from ddgs import DDGS
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from sol.config import settings
from sol.tools.permissions import require_permission

log = structlog.get_logger()


def _search_ddg(query: str, max_results: int) -> list[dict]:
    """Run DuckDuckGo search synchronously."""
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


@tool
@require_permission("web_search", arg="query", display="Search: {query}")
async def web_search(query: str, config: RunnableConfig, max_results: int = 0) -> str:
    """Search the web using DuckDuckGo. Returns a list of results with titles, URLs, and snippets.

    Use this to find current information, documentation, or answers to questions.
    """
    effective_max = max_results if max_results > 0 else settings.tools.web_search_max_results

    log.info("tool.web_search.execute", query=query, max_results=effective_max)
    try:
        results = await asyncio.to_thread(_search_ddg, query, effective_max)
    except Exception as exc:
        return f"[Error] Search failed: {exc}"

    if not results:
        return "No search results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"{i}. [{title}]({href})\n   {body}")
    return "\n\n".join(lines)
