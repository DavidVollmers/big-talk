from typing import Sequence, TypeAlias, AsyncGenerator
from dataclasses import dataclass

from .message import Message, OutputMessage
from .middleware import MiddlewareHandler, Middleware, MiddlewareStack


@dataclass
class StreamResultContext:
    model: str
    iterations: int
    input_messages: Sequence[Message]
    message_history: Sequence[Message]


StreamResultHandler: TypeAlias = MiddlewareHandler[StreamResultContext, AsyncGenerator[OutputMessage, None]]

StreamResultMiddleware: TypeAlias = Middleware[StreamResultContext, AsyncGenerator[OutputMessage, None]]

StreamResultMiddlewareStack: TypeAlias = MiddlewareStack[StreamResultContext, AsyncGenerator[OutputMessage, None]]


class BaseStreamResultHandler(StreamResultHandler):
    async def __call__(self, ctx: StreamResultContext) -> AsyncGenerator[OutputMessage, None]:
        return
        yield
