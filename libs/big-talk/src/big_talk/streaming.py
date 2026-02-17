from dataclasses import dataclass
from typing import Callable, Sequence, TypeAlias, AsyncGenerator, Any

from .middleware import MiddlewareStack, MiddlewareHandler, Middleware
from .tool import Tool
from .llm import LLMProvider
from .message import Message, OutputMessage


@dataclass
class StreamContext:
    model: str
    iteration: int
    tools: Sequence[Tool]
    messages: Sequence[Message]
    _provider_resolver: Callable[[str], tuple[LLMProvider, str]]

    def get_llm_provider(self) -> tuple[LLMProvider, str]:
        return self._provider_resolver(self.model)


StreamHandler: TypeAlias = MiddlewareHandler[StreamContext, AsyncGenerator[OutputMessage, None]]

StreamingMiddleware: TypeAlias = Middleware[StreamContext, AsyncGenerator[OutputMessage, None]]

StreamingMiddlewareStack: TypeAlias = MiddlewareStack[StreamContext, AsyncGenerator[OutputMessage, None]]


class BaseStreamHandler(StreamHandler):
    async def __call__(self, ctx: StreamContext, **kwargs: Any) -> AsyncGenerator[OutputMessage, None]:
        provider, model_name = ctx.get_llm_provider()
        async for message in provider.stream(
                model=model_name,
                messages=ctx.messages,
                tools=ctx.tools,
                **kwargs
        ):
            yield message
