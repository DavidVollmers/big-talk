from abc import ABC
from dataclasses import dataclass
from typing import Callable, TypeAlias, AsyncGenerator, Any

from .middleware import MiddlewareStack, MiddlewareHandler, Middleware
from .tool import Tool
from .llm import LLMProvider
from .message import Message, OutputMessage


@dataclass
class StreamContextBase(ABC):
    model: str
    tools: list[Tool]
    messages: list[Message]
    _provider_resolver: Callable[[str], tuple[LLMProvider, str]]

    def get_llm_provider(self) -> tuple[LLMProvider, str]:
        return self._provider_resolver(self.model)


@dataclass
class StreamIterationContext(StreamContextBase):
    iteration: int


StreamIterationHandler: TypeAlias = MiddlewareHandler[StreamIterationContext, AsyncGenerator[OutputMessage, None]]

StreamIterationMiddleware: TypeAlias = Middleware[StreamIterationContext, AsyncGenerator[OutputMessage, None]]

StreamIterationMiddlewareStack: TypeAlias = MiddlewareStack[StreamIterationContext, AsyncGenerator[OutputMessage, None]]


class BaseStreamIterationHandler(StreamIterationHandler):
    async def __call__(self, ctx: StreamIterationContext, **kwargs: Any) -> AsyncGenerator[OutputMessage, None]:
        provider, model_name = ctx.get_llm_provider()
        async for message in provider.stream(
                model=model_name,
                messages=ctx.messages,
                tools=ctx.tools,
                **kwargs
        ):
            yield message
