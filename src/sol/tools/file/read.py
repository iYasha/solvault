from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from sol.config import settings
from sol.tools.permissions import require_permission
from sol.utils import FileError, FileManager


@tool
@require_permission("file_read", arg="path", display="Read: {path}")
async def file_read(
    path: str,
    config: RunnableConfig,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read the contents of a file. Returns the content with line numbers.

    Optionally specify start_line and end_line to read a specific range.
    """
    try:
        fm = FileManager(path, settings.tools.allowed_paths)
        return await fm.read(start_line, end_line)
    except FileError as exc:
        return f"[Error] {exc}"
