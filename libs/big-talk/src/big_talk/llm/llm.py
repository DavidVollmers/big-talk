from abc import ABC, abstractmethod
from typing import Iterable, AsyncGenerator

from ..message import Message


class LLM(ABC):
    @abstractmethod
    async def count_tokens(self, model: str, messages: Iterable[Message], **kwargs) -> int:
        pass

    @abstractmethod
    async def stream(self, model: str, messages: Iterable[Message], **kwargs) -> AsyncGenerator[Message, None]:
        pass
