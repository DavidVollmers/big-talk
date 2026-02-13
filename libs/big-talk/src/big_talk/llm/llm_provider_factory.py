from typing import Callable

from .llm_provider import LLMProvider

LLMProviderFactory = Callable[[], LLMProvider]
