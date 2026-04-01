from pathlib import Path

import structlog

from sol.config import settings

log = structlog.get_logger()

DEFAULT_SYSTEM_PROMPT = """\
You are Sol, a privacy-first personal AI assistant.

You run locally on the user's machine. All conversations stay private — nothing \
leaves this device unless the user explicitly approves it.

Be helpful, concise, and respectful of the user's time. When you don't know \
something, say so honestly rather than guessing.\
"""


def load_system_prompt() -> str:
    """Load system prompt from configured file, falling back to default."""
    path_str = settings.llm.system_prompt_file
    if not path_str:
        return DEFAULT_SYSTEM_PROMPT

    path = Path(path_str).expanduser()
    if not path.exists():
        log.warning("system_prompt.file_not_found", path=str(path))
        return DEFAULT_SYSTEM_PROMPT

    return path.read_text().strip()
