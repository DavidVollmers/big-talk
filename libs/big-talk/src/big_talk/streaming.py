from dataclasses import dataclass
from typing import Callable, Sequence, TypeAlias, AsyncGenerator, Any

from .middleware import MiddlewareStack, MiddlewareHandler, Middleware
from .tool import Tool
from .llm import LLMProvider
from .message import Message, AssistantMessage


@dataclass
class StreamContext:
    model: str
    tools: Sequence[Tool]
    messages: Sequence[Message]
    _provider_resolver: Callable[[str], tuple[LLMProvider, str]]

    def get_llm_provider(self) -> tuple[LLMProvider, str]:
        return self._provider_resolver(self.model)


StreamHandler: TypeAlias = MiddlewareHandler[StreamContext, AsyncGenerator[AssistantMessage, None]]

StreamingMiddleware: TypeAlias = Middleware[StreamContext, AsyncGenerator[AssistantMessage, None]]

StreamingMiddlewareStack: TypeAlias = MiddlewareStack[StreamContext, AsyncGenerator[AssistantMessage, None]]


class BaseStreamHandler(StreamHandler):
    async def __call__(self, ctx: StreamContext, **kwargs: Any) -> AsyncGenerator[AssistantMessage, None]:
        provider, model_name = ctx.get_llm_provider()
        async for message in provider.stream(
                model=model_name,
                messages=ctx.messages,
                tools=ctx.tools,
                **kwargs
        ):
            yield message
