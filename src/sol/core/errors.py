class AgentError(Exception):
    """Raised when the agent fails to generate a response."""


class ToolError(Exception):
    """Raised when a tool execution fails."""


class PermissionDeniedError(ToolError):
    """Raised when a tool invocation is denied by the permission system."""
