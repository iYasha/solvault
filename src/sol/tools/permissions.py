from __future__ import annotations

import functools
from collections.abc import Callable, Coroutine
from enum import StrEnum
from fnmatch import fnmatch
from typing import Any

import structlog

from sol.config import ToolsConfig

log = structlog.get_logger()


class Permission(StrEnum):
    AUTO_ALLOW = "auto_allow"
    ASK = "ask"
    DENY = "deny"


class PermissionGate:
    """Resolves effective permission for a tool invocation.

    Resolution order: deny globs → allow globs → tool default.
    """

    def __init__(self, config: ToolsConfig) -> None:
        self._config = config

    def resolve(self, tool_name: str, primary_arg: str = "") -> Permission:
        """Return the effective permission for a tool + its primary argument."""
        deny_patterns = self._config.permissions.deny.get(tool_name, [])
        for pattern in deny_patterns:
            if fnmatch(primary_arg, pattern):
                log.debug("permission.deny_glob_matched", tool=tool_name, pattern=pattern)
                return Permission.DENY

        allow_patterns = self._config.permissions.allow.get(tool_name, [])
        for pattern in allow_patterns:
            if fnmatch(primary_arg, pattern):
                log.debug("permission.allow_glob_matched", tool=tool_name, pattern=pattern)
                return Permission.AUTO_ALLOW

        default_str = self._config.permissions.defaults.get(tool_name, "ask")
        return Permission(default_str)


def require_permission(
    tool_name: str,
    *,
    arg: str,
    display: str,
) -> Callable[..., Callable[..., Coroutine[Any, Any, str]]]:
    """Decorator that checks permissions before executing a tool function.

    Reads ``gate`` and ``approval_callback`` from the LangChain
    ``RunnableConfig["configurable"]`` that LangGraph passes to every tool.

    Args:
        tool_name: The tool identifier for permission lookup.
        arg: Name of the parameter to use as the primary argument for glob matching.
        display: Format string for the approval prompt, e.g. "Run: {command}".

    Usage::

        @tool
        @require_permission("shell", arg="command", display="Run: {command}")
        async def shell(command: str, config: RunnableConfig) -> str:
            ...
    """

    def decorator(
        fn: Callable[..., Coroutine[Any, Any, str]],
    ) -> Callable[..., Coroutine[Any, Any, str]]:
        @functools.wraps(fn)
        async def wrapper(*args: object, **kwargs: object) -> str:  # noqa: ANN401
            config = kwargs.get("config")
            configurable = config.get("configurable", {}) if config else {}  # type: ignore[union-attr]
            gate: PermissionGate | None = configurable.get("gate")
            callback = configurable.get("approval_callback")

            if gate and callback:
                primary_arg = str(kwargs.get(arg, ""))
                display_str = display.format(**kwargs)
                perm = gate.resolve(tool_name, primary_arg)
                if perm == Permission.DENY:
                    return f"[Permission denied] {tool_name} denied for: {primary_arg}"
                if perm == Permission.ASK:
                    approved = await callback.request(tool_name, display_str)
                    if not approved:
                        return f"[Permission denied] User denied {tool_name}: {primary_arg}"

            return await fn(*args, **kwargs)

        return wrapper

    return decorator
