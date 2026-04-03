import asyncio

import structlog
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from sol.config import settings
from sol.tools.permissions import require_permission

log = structlog.get_logger()

_MAX_OUTPUT_CHARS = 8000


@tool
@require_permission("shell", arg="command", display="Run: {command}")
async def shell(command: str, config: RunnableConfig) -> str:
    """Execute a shell command and return its output.

    Use for running programs, scripts, git commands, or system commands.
    The command runs with a timeout and returns combined stdout/stderr.
    """
    timeout = settings.tools.shell_timeout

    log.info("tool.shell.execute", command=command, timeout=timeout)
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        return f"[Error] Command timed out after {timeout}s"
    except OSError as exc:
        return f"[Error] Failed to run command: {exc}"

    output = stdout.decode(errors="replace") if stdout else "(no output)"
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(output)} total chars)"
    return output
