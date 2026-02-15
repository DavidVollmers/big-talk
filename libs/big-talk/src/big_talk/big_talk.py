import asyncio
from typing import Sequence, Any, AsyncGenerator, Callable

from .middleware import MiddlewareStack
from .streaming import StreamContext, StreamingMiddlewareStack, BaseStreamHandler
from .tool import Tool
from .llm import LLMProvider, LLMProviderFactory
from .message import Message, AssistantMessage, SystemMessage, ToolUse
from .tool_execution import ToolExecutionMiddlewareStack, BaseToolExecutionHandler, ToolExecutionContext


class BigTalk:
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._provider_factories: dict[str, LLMProviderFactory] = {
            'anthropic': self._anthropic_provider_factory,
            'openai': self._openai_provider_factory,
        }
        self._streaming: StreamingMiddlewareStack = MiddlewareStack(BaseStreamHandler())
        self._tool_execution: ToolExecutionMiddlewareStack = MiddlewareStack(BaseToolExecutionHandler())

    @property
    def streaming(self) -> StreamingMiddlewareStack:
        return self._streaming

    @property
    def tool_execution(self) -> ToolExecutionMiddlewareStack:
        return self._tool_execution

    def add_provider(self, name: str, provider_factory: LLMProviderFactory, override: bool = False) -> None:
        if not override and (name in self._providers or name in self._provider_factories):
            raise ValueError(f'Provider "{name}" is already registered.')
        self._provider_factories[name] = provider_factory
        if name in self._providers:
            del self._providers[name]

    @staticmethod
    def _parse_model(model: str) -> tuple[str, str]:
        if '/' not in model:
            raise ValueError(f'Invalid model name: {model}. Expected format: "provider/model_name".')
        provider, model_name = model.split('/', 1)
        return provider, model_name

    def _get_provider(self, provider: str) -> LLMProvider:
        if provider in self._providers:
            return self._providers[provider]
        elif provider in self._provider_factories:
            llm = self._provider_factories[provider]()
            self._providers[provider] = llm
            return llm
        else:
            raise NotImplementedError(f'Provider "{provider}" is not supported.')

    def _get_llm_provider(self, model: str) -> tuple[LLMProvider, str]:
        provider, model_name = self._parse_model(model)
        return self._get_provider(provider), model_name

    async def stream(self,
                     model: str,
                     messages: Sequence[Message],
                     tools: Sequence[Callable | Tool] = None,
                     max_iterations: int = 10,
                     **kwargs: Any) -> AsyncGenerator[AssistantMessage, None]:
        if not any(message['role'] == 'user' for message in messages):
            raise ValueError('At least one user message is required to generate a response.')

        normalized_tools = self._normalize_tools(tools)

        current_history = list(messages)

        stream_handler = self._streaming.build()
        tool_execution_handler = self._tool_execution.build()

        for iteration in range(max_iterations):
            stream_ctx = StreamContext(model=model, tools=normalized_tools, messages=current_history,
                                       _provider_resolver=self._get_llm_provider)

            tool_uses: list[ToolUse] = []
            async for message in stream_handler(stream_ctx, **kwargs):
                yield message

                if not message['is_aggregate']:
                    continue

                current_history.append(message)

                tool_uses.extend([b for b in message['content'] if b['type'] == 'tool_use'])

            if not tool_uses:
                break

            tool_execution_ctx = ToolExecutionContext(
                tool_uses=tool_uses,
                tools=normalized_tools,
                messages=current_history
            )

            tool_results = await asyncio.gather(*tool_execution_handler(tool_execution_ctx))

            current_history.append(SystemMessage(
                role='system',
                content=tool_results
            ))

    async def close(self):
        results = await asyncio.gather(*(provider.close() for provider in self._providers.values()),
                                       return_exceptions=True)
        exceptions = [result for result in results if isinstance(result, Exception)]
        if exceptions:
            raise ExceptionGroup('One or more providers failed to close', exceptions)

    @staticmethod
    def _normalize_tools(tools: Sequence[Callable | Tool] | None) -> list[Tool]:
        normalized = []

        if not tools:
            return normalized

        for t in tools:
            if isinstance(t, Tool):
                normalized.append(t)
            elif callable(t):
                normalized.append(Tool.from_func(t))
            else:
                raise TypeError(f'Invalid tool type: {type(t)}')

        return normalized

    @staticmethod
    def _anthropic_provider_factory() -> LLMProvider:
        from .llm.anthropic import AnthropicProvider
        return AnthropicProvider()

    @staticmethod
    def _openai_provider_factory() -> LLMProvider:
        from .llm.openai import OpenAIProvider
        return OpenAIProvider()
