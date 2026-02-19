from dataclasses import dataclass
from typing import TypeAlias, AsyncGenerator
from uuid import uuid4

from .loop import extract_tool_uses, use_tools
from .tool_execution import ToolExecutionHandler
from .middleware import MiddlewareHandler, Middleware, MiddlewareStack
from .stream_iteration import StreamIterationContext, StreamContextBase, StreamIterationHandler
from .message import Message, ToolUse, ToolMessage


@dataclass
class StreamContext(StreamContextBase):
    max_iterations: int
    _stream_iteration_handler: StreamIterationHandler
    _tool_execution_handler: ToolExecutionHandler


StreamHandler: TypeAlias = MiddlewareHandler[StreamContext, AsyncGenerator[Message, None]]

StreamMiddleware: TypeAlias = Middleware[StreamContext, AsyncGenerator[Message, None]]

StreamMiddlewareStack: TypeAlias = MiddlewareStack[StreamContext, AsyncGenerator[Message, None]]


class BaseStreamHandler(StreamHandler):
    async def __call__(self, ctx: StreamContext, **kwargs) -> AsyncGenerator[Message, None]:
        current_history = list(ctx.messages)
        for iteration in range(ctx.max_iterations):
            # noinspection PyProtectedMember
            stream_ctx = StreamIterationContext(model=ctx.model, tools=ctx.tools, messages=current_history,
                                                _provider_resolver=ctx._provider_resolver, iteration=iteration)

            tool_uses_by_parent: list[tuple[str, ToolUse]] = []
            # noinspection PyProtectedMember
            async for message in ctx._stream_iteration_handler(stream_ctx, **kwargs):
                yield message

                is_app_message = message['role'] == 'app'
                if not is_app_message and not message.get('is_aggregate'):
                    continue

                current_history.append(message)

                if is_app_message:
                    continue

                tool_uses_by_parent.extend(extract_tool_uses(message))

            if not tool_uses_by_parent:
                break

            # noinspection PyProtectedMember
            results_by_parent = await use_tools(tool_uses_by_parent, current_history, ctx.tools, iteration,
                                                ctx._tool_execution_handler)

            for parent_id, results in results_by_parent.items():
                tool_result_message = ToolMessage(
                    id=str(uuid4()),
                    role='tool',
                    content=results,
                    parent_id=parent_id,
                )

                yield tool_result_message
                current_history.append(tool_result_message)
