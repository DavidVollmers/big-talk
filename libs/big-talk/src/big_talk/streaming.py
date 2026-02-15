from dataclasses import dataclass
from typing import Callable, Sequence, TypeAlias, AsyncGenerator, Any

from .middleware import MiddlewareStack, MiddlewareHandler
from .tool import Tool
from .llm import LLMProvider
from .message import Message


@dataclass
class StreamContext:
    model: str
    tools: Sequence[Tool]
    messages: Sequence[Message]
    _provider_resolver: Callable[[str], tuple[LLMProvider, str]]

    def get_llm_provider(self) -> tuple[LLMProvider, str]:
        return self._provider_resolver(self.model)


StreamingMiddlewareStack: TypeAlias = MiddlewareStack[StreamContext, AsyncGenerator[Message, None]]


class BaseStreamHandler(MiddlewareHandler[StreamContext, AsyncGenerator[Message, None]]):
    async def __call__(self, ctx: StreamContext, **kwargs: Any) -> AsyncGenerator[Message, None]:
        provider, model_name = ctx.get_llm_provider()
        async for message in provider.stream(
                model=model_name,
                messages=ctx.messages,
                tools=ctx.tools,
                **kwargs
        ):
            yield message
