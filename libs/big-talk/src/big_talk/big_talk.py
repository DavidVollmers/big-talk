import asyncio
from collections.abc import AsyncGenerator
from typing import Iterable

from .llm import LLMProvider
from .message import Message


class BigTalk:
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}

    @staticmethod
    def _parse_model(model: str) -> tuple[str, str]:
        if '/' not in model:
            raise ValueError(f'Invalid model name: {model}. Expected format: "provider/model_name".')
        provider, model_name = model.split('/', 1)
        return provider, model_name

    def _resolve_provider(self, provider: str) -> LLMProvider:
        if provider in self._providers:
            return self._providers[provider]
        elif provider == 'anthropic':
            from .llm.anthropic_provider import AnthropicProvider
            llm = AnthropicProvider()
            self._providers[provider] = llm
            return llm
        else:
            raise NotImplementedError(f'Provider "{provider}" is not supported.')

    async def stream(self, model: str, messages: Iterable[Message], **kwargs) -> AsyncGenerator[Message, None]:
        provider, model_name = self._parse_model(model)
        llm = self._resolve_provider(provider)
        async for message in await llm.stream(model=model_name, messages=messages, **kwargs):
            yield message

    async def close(self):
        results = await asyncio.gather(*(provider.close() for provider in self._providers.values()),
                                       return_exceptions=True)
        exceptions = [result for result in results if isinstance(result, Exception)]
        if exceptions:
            raise ExceptionGroup('One or more providers failed to close', exceptions)
