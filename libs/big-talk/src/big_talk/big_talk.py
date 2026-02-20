import asyncio
from typing import Sequence, Any, AsyncGenerator, Callable, Iterable
from uuid import uuid4

from .loop import extract_tool_uses, use_tools
from .middleware import MiddlewareStack
from .stream import StreamMiddlewareStack, BaseStreamHandler, StreamContext
from .stream_iteration import StreamIterationMiddlewareStack, BaseStreamIterationHandler
from .tool import Tool
from .llm import LLMProvider, LLMProviderFactory
from .message import Message, ToolUse, ToolMessage
from .tool_execution import ToolExecutionMiddlewareStack, BaseToolExecutionHandler, ToolExecutionContext

DEFAULT_MAX_ITERATIONS = 10


class BigTalk:
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._provider_factories: dict[str, LLMProviderFactory] = {
            'anthropic': self._anthropic_provider_factory,
            'openai': self._openai_provider_factory,
        }
        self._stream_iteration: StreamIterationMiddlewareStack = MiddlewareStack(BaseStreamIterationHandler())
        self._tool_execution: ToolExecutionMiddlewareStack = MiddlewareStack(BaseToolExecutionHandler())
        self._streaming: StreamMiddlewareStack = MiddlewareStack(BaseStreamHandler())

    @property
    def streaming(self) -> StreamMiddlewareStack:
        return self._streaming

    @property
    def stream_iteration(self) -> StreamIterationMiddlewareStack:
        return self._stream_iteration

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

    async def send(self, model: str, messages: Sequence[Message], tools: Sequence[Callable | Tool] = None,
                   max_iterations: int = DEFAULT_MAX_ITERATIONS, **kwargs: Any) -> Iterable[Message]:
        if not any(message['role'] == 'user' for message in messages):
            raise ValueError('At least one user message is required to generate a response.')

        normalized_tools = self._normalize_tools(tools)

        tool_execution_handler = self._tool_execution.build()

        provider, model_name = self._get_llm_provider(model)

        current_history = list(messages)
        for iteration in range(max_iterations):
            message = await provider.send(model_name, current_history, tools=normalized_tools, **kwargs)

            current_history.append(message)

            tool_uses_by_parent = extract_tool_uses(message)

            if not tool_uses_by_parent:
                break

            results_by_parent = await use_tools(tool_uses_by_parent, current_history, normalized_tools, iteration,
                                                tool_execution_handler)

            for parent_id, results in results_by_parent.items():
                tool_result_message = ToolMessage(
                    id=str(uuid4()),
                    role='tool',
                    content=results,
                    parent_id=parent_id,
                )

                current_history.append(tool_result_message)

        return current_history[len(messages):]

    async def stream(self,
                     model: str,
                     messages: Sequence[Message],
                     tools: Sequence[Callable | Tool] = None,
                     max_iterations: int = DEFAULT_MAX_ITERATIONS,
                     **kwargs: Any) -> AsyncGenerator[Message, None]:
        if not any(message['role'] == 'user' for message in messages):
            raise ValueError('At least one user message is required to generate a response.')

        normalized_tools = self._normalize_tools(tools)

        stream_iteration_handler = self._stream_iteration.build()
        tool_execution_handler = self._tool_execution.build()
        streaming_handler = self._streaming.build()

        ctx = StreamContext(
            model=model,
            tools=normalized_tools,
            messages=list(messages),
            _provider_resolver=self._get_llm_provider,
            max_iterations=max_iterations,
            _stream_iteration_handler=stream_iteration_handler,
            _tool_execution_handler=tool_execution_handler
        )

        async for message in streaming_handler(ctx, **kwargs):
            yield message

    async def execute_tool(self, tool: Callable | Tool, params: dict[str, Any],
                           messages: Sequence[Message] = None, metadata: dict[str, Any] = None) -> Any | None:
        normalized_tool = self._normalize_tools([tool])[0]

        tool_use = ToolUse(
            type='tool_use',
            id=str(uuid4()),
            name=normalized_tool.name,
            params=params,
            metadata=metadata
        )

        context = ToolExecutionContext(
            tool_uses=[tool_use],
            tools=[normalized_tool],
            messages=messages or [],
            iteration=0
        )

        handler = self._tool_execution.build()
        tasks = await handler(context)
        results = await asyncio.gather(*tasks)

        if not results:
            return None

        result = results[0]
        if result['is_error']:
            raise Exception(f'Tool execution failed: {result['result']}')
        return result['result']

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
