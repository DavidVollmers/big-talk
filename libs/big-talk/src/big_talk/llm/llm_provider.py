from abc import ABC, abstractmethod
from typing import Sequence, AsyncGenerator, Iterable

from ..tool import Tool
from ..message import Message, AssistantMessage


class LLMProvider(ABC):
    @abstractmethod
    async def count_tokens(self, model: str, messages: Sequence[Message], tools: Sequence[Tool], **kwargs) -> int:
        pass

    @abstractmethod
    async def send(self, model: str, messages: Sequence[Message], tools: Sequence[Tool], **kwargs) -> AssistantMessage:
        pass

    @abstractmethod
    def stream(self, model: str, messages: Sequence[Message], tools: Sequence[Tool], **kwargs) \
            -> AsyncGenerator[AssistantMessage, None]:
        pass

    async def close(self):
        pass
