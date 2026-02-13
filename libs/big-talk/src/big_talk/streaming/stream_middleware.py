from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any, Callable, Union

from ..message import Message
from .stream_context import StreamContext
from .stream_handler import StreamHandler

CallableStreamMiddleware = Callable[..., AsyncGenerator[Message, None]]


class StreamMiddlewareBase(ABC):
    @abstractmethod
    def __call__(self, handler: StreamHandler, ctx: StreamContext, **kwargs: Any) -> AsyncGenerator[Message, None]:
        pass


StreamMiddleware = Union[CallableStreamMiddleware, StreamMiddlewareBase]
