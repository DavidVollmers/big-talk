from dataclasses import dataclass
from typing import Sequence, TypeAlias, AsyncIterable

from .message import ToolUse, Message, ToolResult
from .middleware import MiddlewareStack, MiddlewareHandler
from .tool import Tool


@dataclass
class ToolExecutionContext:
    tool_uses: Sequence[ToolUse]
    tools: Sequence[Tool]
    messages: Sequence[Message]


ToolExecutionHandler: TypeAlias = MiddlewareHandler[ToolExecutionContext, AsyncIterable[ToolResult]]

ToolExecutionMiddlewareStack: TypeAlias = MiddlewareStack[ToolExecutionContext, AsyncIterable[ToolResult]]


class BaseToolExecutionHandler(ToolExecutionHandler):
    async def __call__(self, context: ToolExecutionContext) -> AsyncIterable[ToolResult]:
        raise NotImplementedError()
