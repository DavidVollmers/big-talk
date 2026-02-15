import inspect
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
        tool_map = {tool.name: tool for tool in context.tools}

        tasks: list[Awaitable[ToolResult]] = []
        for tool_use in context.tool_uses:
            if tool_use['name'] not in tool_map:
                tasks.append(self._error_result(tool_use['id'], f'Tool {tool_use["name"]} not found'))
                continue

            tool = tool_map[tool_use['name']]
            tasks.append(self._execute_tool(tool, tool_use))

        return tasks

    @staticmethod
    async def _error_result(tool_use_id: str, error_message: str) -> ToolResult:
        return ToolResult(
            type='tool_result',
            tool_use_id=tool_use_id,
            result=error_message,
            is_error=True
        )

    @staticmethod
    async def _execute_tool(tool: Tool, tool_use: ToolUse) -> ToolResult:
        try:
            if inspect.iscoroutinefunction(tool.func):
                result = await tool.func(**tool_use['params'])
            else:
                result = tool.func(**tool_use['params'])
            return ToolResult(
                type='tool_result',
                tool_use_id=tool_use['id'],
                result=result,
                is_error=False
            )
        except Exception as e:
            return ToolResult(
                type='tool_result',
                tool_use_id=tool_use['id'],
                result=str(e),
                is_error=True
            )
