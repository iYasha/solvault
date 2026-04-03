from sol.tools.approval import ApprovalCallback, DenyAllApprovalCallback, WebSocketApprovalCallback
from sol.tools.file.edit import file_edit
from sol.tools.file.read import file_read
from sol.tools.file.write import file_write
from sol.tools.memory.save import memory_save
from sol.tools.memory.search import memory_search
from sol.tools.permissions import Permission, PermissionGate
from sol.tools.shell import shell
from sol.tools.web.fetch import web_fetch
from sol.tools.web.research import web_research
from sol.tools.web.search import web_search

ALL_TOOLS = [shell, file_read, file_write, file_edit, web_search, web_fetch, web_research, memory_search, memory_save]

__all__ = [
    "ALL_TOOLS",
    "ApprovalCallback",
    "DenyAllApprovalCallback",
    "Permission",
    "PermissionGate",
    "WebSocketApprovalCallback",
    "file_edit",
    "file_read",
    "file_write",
    "memory_save",
    "memory_search",
    "shell",
    "web_fetch",
    "web_research",
    "web_search",
]
