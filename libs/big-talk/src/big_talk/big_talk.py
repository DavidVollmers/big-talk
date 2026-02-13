import asyncio
from collections.abc import AsyncGenerator
from typing import Iterable, Any

from .llm import LLMProvider, LLMProviderFactory
from .message import Message
from .stream_middleware import StreamMiddleware, StreamHandler


class MiddlewareWrapper(StreamHandler):
    def __init__(self, mw: StreamMiddleware, next_handler: StreamHandler):
        self._mw = mw
        self._next_handler = next_handler

    async def __call__(self, model: str, messages: Iterable[Message], **kwargs: Any) -> AsyncGenerator[
        Message, None]:
        async for message in self._mw(self._next_handler, model, messages, **kwargs):
            yield message


class BigTalk:
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._provider_factories: dict[str, LLMProviderFactory] = {
            'anthropic': self._anthropic_provider_factory
        }
        self._middleware: list[StreamMiddleware] = []

    def add_middleware(self, middleware: StreamMiddleware):
        self._middleware.append(middleware)

    def add_provider(self, name: str, provider_factory: LLMProviderFactory):
        if name in self._providers or name in self._provider_factories:
            raise ValueError(f'Provider "{name}" is already registered.')
        self._provider_factories[name] = provider_factory

    @staticmethod
    def _parse_model(model: str) -> tuple[str, str]:
        if '/' not in model:
            raise ValueError(f'Invalid model name: {model}. Expected format: "provider/model_name".')
        provider, model_name = model.split('/', 1)
        return provider, model_name

    def _resolve_provider(self, provider: str) -> LLMProvider:
        if provider in self._providers:
            return self._providers[provider]
        elif provider in self._provider_factories:
            llm = self._provider_factories[provider]()
            self._providers[provider] = llm
            return llm
        else:
            raise NotImplementedError(f'Provider "{provider}" is not supported.')

    async def stream(self, model: str, messages: Iterable[Message], **kwargs: Any) -> AsyncGenerator[Message, None]:
        handler = self._build_middleware_stack()
        response = handler(model, messages, **kwargs)
        async for message in response:
            yield message

    async def close(self):
        results = await asyncio.gather(*(provider.close() for provider in self._providers.values()),
                                       return_exceptions=True)
        exceptions = [result for result in results if isinstance(result, Exception)]
        if exceptions:
            raise ExceptionGroup('One or more providers failed to close', exceptions)

    async def _llm_stream(self, model: str, messages: Iterable[Message], **kwargs: Any) -> AsyncGenerator[
        Message, None]:
        provider, model_name = self._parse_model(model)
        llm = self._resolve_provider(provider)

        async for message in llm.stream(model=model_name, messages=messages, **kwargs):
            yield message

    def _build_middleware_stack(self) -> StreamHandler:
        handler = self._llm_stream

        for middleware in reversed(self._middleware):
            handler = MiddlewareWrapper(middleware, handler)

        return handler

    @staticmethod
    def _anthropic_provider_factory() -> LLMProvider:
        from .llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider()
