from typing import AsyncGenerator, Protocol, runtime_checkable, Any

from ..message import Message
from .stream_context import StreamContext
from .stream_handler import StreamHandler


@runtime_checkable
class StreamMiddleware(Protocol):
    def __call__(self, handler: StreamHandler, ctx: StreamContext, **kwargs: Any) -> AsyncGenerator[Message, None]: ...
