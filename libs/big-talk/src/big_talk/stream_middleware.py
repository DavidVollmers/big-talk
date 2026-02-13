from abc import ABC, abstractmethod
from typing import Iterable, AsyncGenerator, Protocol, runtime_checkable, Any

from .message import Message


class StreamHandler(ABC):
    @abstractmethod
    def __call__(self, model: str, messages: Iterable[Message], **kwargs: Any) -> AsyncGenerator[Message, None]: ...


@runtime_checkable
class StreamMiddleware(Protocol):
    def __call__(self, handler: StreamHandler, model: str, messages: Iterable[Message], **kwargs: Any) -> \
            AsyncGenerator[Message, None]: ...
