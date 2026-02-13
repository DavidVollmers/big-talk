from abc import ABC, abstractmethod
from typing import Sequence, AsyncGenerator

from .message import Message


class LLMProvider(ABC):
    @abstractmethod
    async def count_tokens(self, model: str, messages: Sequence[Message], **kwargs) -> int:
        pass

    @abstractmethod
    async def stream(self, model: str, messages: Sequence[Message], **kwargs) -> AsyncGenerator[Message, None]:
        pass
