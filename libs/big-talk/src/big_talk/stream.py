import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import TypeAlias, AsyncGenerator
from uuid import uuid4

from . import ToolExecutionContext, ToolExecutionHandler
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

                parent_id = message['id']

                tool_uses = [b for b in message['content'] if b['type'] == 'tool_use']
                for tool_use in tool_uses:
                    tool_uses_by_parent.append((parent_id, tool_use))

            if not tool_uses_by_parent:
                break

            tool_uses = [tu for _, tu in tool_uses_by_parent]

            tool_execution_ctx = ToolExecutionContext(
                tool_uses=tool_uses,
                tools=ctx.tools,
                messages=current_history,
                iteration=iteration
            )

            # noinspection PyProtectedMember
            tool_tasks = await ctx._tool_execution_handler(tool_execution_ctx)
            tool_results = await asyncio.gather(*tool_tasks)

            results_by_parent = defaultdict(list)
            for (parent_id, _), result in zip(tool_uses_by_parent, tool_results):
                results_by_parent[parent_id].append(result)

            for parent_id, results in results_by_parent.items():
                tool_result_message = ToolMessage(
                    id=str(uuid4()),
                    role='tool',
                    content=results,
                    parent_id=parent_id,
                )

                yield tool_result_message
                current_history.append(tool_result_message)
