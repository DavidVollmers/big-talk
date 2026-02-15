from dataclasses import dataclass
from typing import Sequence, TypeAlias, Iterable, Awaitable

from .message import ToolUse, Message, ToolResult
from .middleware import MiddlewareStack, MiddlewareHandler
from .tool import Tool


@dataclass
class ToolExecutionContext:
    tool_uses: Sequence[ToolUse]
    tools: Sequence[Tool]
    messages: Sequence[Message]


ToolExecutionHandler: TypeAlias = MiddlewareHandler[ToolExecutionContext, Iterable[Awaitable[ToolResult]]]

ToolExecutionMiddlewareStack: TypeAlias = MiddlewareStack[ToolExecutionContext, Iterable[Awaitable[ToolResult]]]


class BaseToolExecutionHandler(ToolExecutionHandler):
    async def __call__(self, context: ToolExecutionContext) -> Iterable[Awaitable[ToolResult]]:
        raise NotImplementedError()
