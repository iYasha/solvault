from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from sol.config import settings
from sol.tools.permissions import require_permission
from sol.utils import FileError, FileManager


@tool
@require_permission("file_write", arg="path", display="Write {path}")
async def file_write(path: str, content: str, config: RunnableConfig) -> str:
    """Create or overwrite a file with the given content.

    Creates parent directories if needed. Use file_edit for modifying existing files.
    """
    try:
        fm = FileManager(path, settings.tools.allowed_paths)
        return await fm.write(content)
    except FileError as exc:
        return f"[Error] {exc}"
