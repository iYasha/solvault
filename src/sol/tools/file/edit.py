from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from sol.config import settings
from sol.tools.permissions import require_permission
from sol.utils import FileError, FileManager


@tool
@require_permission("file_edit", arg="path", display="Edit: {path}")
async def file_edit(
    path: str,
    old_string: str,
    new_string: str,
    config: RunnableConfig,
    replace_all: bool = False,
) -> str:
    """Edit a file by replacing an exact string match.

    Provide old_string (must be unique in the file) and new_string.
    The edit fails if old_string is not found or appears more than once.
    Set replace_all=true to replace all occurrences.
    """
    try:
        fm = FileManager(path, settings.tools.allowed_paths)
        return await fm.edit(old_string, new_string, replace_all)
    except FileError as exc:
        return f"[Error] {exc}"
