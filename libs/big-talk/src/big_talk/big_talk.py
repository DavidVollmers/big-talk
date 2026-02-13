from collections.abc import AsyncGenerator
from typing import Iterable

from .message import Message


class BigTalk:
    async def stream(self, model: str, messages: Iterable[Message]) -> AsyncGenerator[Message, None]:
        raise NotImplementedError()
