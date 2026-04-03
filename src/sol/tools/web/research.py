import asyncio
import json

import structlog
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from sol.config import settings
from sol.tools.permissions import require_permission

log = structlog.get_logger()

_RESEARCH_PROMPT_PREFIX = (
    "You are a research assistant. Use your web search capabilities to answer the following question. "
    "Search multiple sources, cross-check facts, and provide a comprehensive answer. "
    "Include specific details, names, dates, and URLs where relevant. "
    "Be thorough but concise.\n\n"
    "Question: "
)


@tool
@require_permission("web_research", arg="query", display="Research: {query}")
async def web_research(query: str, config: RunnableConfig) -> str:
    """Research a question using Claude with web search capabilities.

    Use this for factual questions that need up-to-date information from the internet,
    such as finding specific details about movies, people, events, products, or current news.
    Claude will search multiple sources and provide a comprehensive answer.
    Prefer this over web_search + web_fetch for complex research questions.
    """
    claude_config = settings.tools.claude
    log.info("tool.web_research.execute", query=query)

    cmd = [
        "claude",
        "-p",
        _RESEARCH_PROMPT_PREFIX + query,
        "--output-format",
        "json",
        "--model",
        claude_config.model,
        "--tools",
        "WebSearch,WebFetch",
        "--allowedTools",
        "WebSearch,WebFetch",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=claude_config.timeout)
    except TimeoutError:
        return f"[Error] Claude CLI timed out after {claude_config.timeout}s"
    except FileNotFoundError:
        return "[Error] Claude CLI not found. Install it with: npm install -g @anthropic-ai/claude-code"
    except OSError as exc:
        return f"[Error] Failed to run Claude CLI: {exc}"

    if proc.returncode != 0:
        error_text = stderr.decode(errors="replace")[:500] if stderr else "Unknown error"
        return f"[Error] Claude CLI exited with code {proc.returncode}: {error_text}"

    return _parse_output(stdout.decode(errors="replace"))


def _parse_output(raw: str) -> str:
    """Parse Claude CLI JSON output, extracting the result text."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip() if raw.strip() else "[Error] Empty response from Claude CLI"

    result = data.get("result") or data.get("content") or ""
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        result = "\n".join(parts)

    cost = data.get("total_cost_usd")
    if cost:
        log.info("tool.web_research.complete", cost_usd=cost)

    return str(result).strip() if result else "[Error] Empty response from Claude CLI"
