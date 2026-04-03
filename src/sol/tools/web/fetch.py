import asyncio
import re

import structlog
import trafilatura
from curl_cffi.requests import AsyncSession
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from sol.config import settings
from sol.tools.permissions import require_permission

log = structlog.get_logger()

_FETCH_TIMEOUT = 30


def _extract_content(html: str) -> str:
    """Extract main content from HTML using trafilatura, with regex fallback."""
    text = trafilatura.extract(html)
    if text:
        return text

    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@tool
@require_permission("web_fetch", arg="url", display="Fetch: {url}")
async def web_fetch(url: str, config: RunnableConfig, extract_text: bool = True) -> str:
    """Fetch the contents of a URL. By default, extracts the main content from the page.

    Set extract_text=false to get raw HTML. Useful for reading documentation, articles, or API responses.
    """
    max_chars = settings.tools.web_fetch_max_chars

    log.info("tool.web_fetch.execute", url=url)
    try:
        async with AsyncSession(impersonate="chrome") as session:
            resp = await session.get(url, timeout=_FETCH_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
    except Exception as exc:
        return f"[Error] Failed to fetch URL: {exc}"

    html = resp.text
    if extract_text:
        content = await asyncio.to_thread(_extract_content, html)
    else:
        content = html

    if not content:
        return (
            "[Error] Page returned no readable text content (likely a JavaScript-rendered site). "
            "Try a different source — use web_search to find an alternative URL with the same information."
        )

    if len(content) > max_chars:
        content = content[:max_chars] + f"\n... (truncated, {len(content)} total chars)"
    return content
