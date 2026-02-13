from abc import abstractmethod, ABC
from typing import Any, AsyncGenerator

from .message import Message
from .stream_context import StreamContext


class StreamHandler(ABC):
    @abstractmethod
    def __call__(self, ctx: StreamContext, **kwargs: Any) -> AsyncGenerator[Message, None]: ...
