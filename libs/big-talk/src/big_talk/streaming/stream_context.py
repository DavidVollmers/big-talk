from dataclasses import dataclass
from typing import Callable, Sequence

from ..tool import Tool
from ..llm import LLMProvider
from ..message import Message


@dataclass
class StreamContext:
    model: str
    tools: Sequence[Tool]
    messages: Sequence[Message]
    _provider_resolver: Callable[[str], tuple[LLMProvider, str]]

    def get_llm_provider(self) -> tuple[LLMProvider, str]:
        return self._provider_resolver(self.model)
