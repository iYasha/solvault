import asyncio
from fnmatch import fnmatch
from pathlib import Path

import structlog

log = structlog.get_logger()


class FileError(Exception):
    """Raised when a file operation fails."""


class FileManager:
    """File system operations with path validation."""

    def __init__(self, path: str, allowed_patterns: list[str], max_read_chars: int = 50_000) -> None:
        self.resolved = Path(path).expanduser().resolve()
        self.allowed_patterns = allowed_patterns
        self.max_read_chars = max_read_chars
        self._validate()

    def _validate(self) -> None:
        """Validate the path is within allowed patterns."""
        resolved_str = str(self.resolved)
        if not any(fnmatch(resolved_str, str(Path(p).expanduser())) for p in self.allowed_patterns):
            raise FileError(f"Path not in allowed paths: {self.resolved}")

    async def read(self, start_line: int | None = None, end_line: int | None = None) -> str:
        """Read the file and return numbered lines, optionally sliced."""
        self._ensure_exists()

        log.info("file_manager.read", path=str(self.resolved))
        try:
            content = await asyncio.to_thread(self.resolved.read_text, encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise FileError(f"Failed to read file: {exc}") from exc

        return self.format_numbered(content, start_line, end_line)

    async def write(self, content: str) -> str:
        """Write content to the file, creating parent directories."""
        log.info("file_manager.write", path=str(self.resolved), chars=len(content))
        try:
            await asyncio.to_thread(self._write_sync, content)
        except OSError as exc:
            raise FileError(f"Failed to write file: {exc}") from exc

        return f"Successfully wrote {len(content)} chars to {self.resolved}"

    async def edit(self, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Edit the file by replacing an exact string match."""
        self._ensure_exists()

        try:
            content = await asyncio.to_thread(self.resolved.read_text, encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise FileError(f"Failed to read file: {exc}") from exc

        if old_string == new_string:
            raise FileError("old_string and new_string are identical")

        count = content.count(old_string)
        if count == 0:
            raise FileError(f"old_string not found in {self.resolved}")
        if count > 1 and not replace_all:
            raise FileError(
                f"old_string is ambiguous — found {count} occurrences in {self.resolved}. Provide more context.",
            )

        new_content = (
            content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        )

        log.info("file_manager.edit", path=str(self.resolved))
        try:
            await asyncio.to_thread(self.resolved.write_text, new_content, encoding="utf-8")
        except OSError as exc:
            raise FileError(f"Failed to write file: {exc}") from exc

        return f"Successfully edited {self.resolved}"

    def format_numbered(self, content: str, start_line: int | None, end_line: int | None) -> str:
        """Format file content with line numbers and optional range slicing."""
        lines = content.splitlines(keepends=True)
        if start_line or end_line:
            start = (start_line or 1) - 1
            end = end_line or len(lines)
            lines = lines[start:end]

        first_num = start_line or 1
        numbered = "".join(f"{i + first_num}\t{line}" for i, line in enumerate(lines))
        if len(numbered) > self.max_read_chars:
            numbered = numbered[: self.max_read_chars] + f"\n... (truncated, {len(content)} total chars)"
        return numbered

    def _ensure_exists(self) -> None:
        """Raise FileError if the path doesn't exist or isn't a file."""
        if not self.resolved.exists():
            raise FileError(f"File not found: {self.resolved}")
        if not self.resolved.is_file():
            raise FileError(f"Not a file: {self.resolved}")

    def _write_sync(self, content: str) -> None:
        self.resolved.parent.mkdir(parents=True, exist_ok=True)
        self.resolved.write_text(content, encoding="utf-8")
