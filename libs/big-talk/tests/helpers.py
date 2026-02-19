import asyncio
from typing import AsyncGenerator, Sequence
from big_talk.llm import LLMProvider
from big_talk import Message, AssistantMessage, Text, Tool
from big_talk.stream_iteration import StreamIterationContext, StreamIterationHandler


class MockToolProvider(LLMProvider):
    def __init__(self, responses: list[AssistantMessage]):
        self.responses = responses

    async def count_tokens(self, model: str, messages: Sequence[Message], **kwargs) -> int:
        pass

    async def send(self, model: str, messages: Sequence[Message], tools: Sequence[Tool], **kwargs) -> AssistantMessage:
        pass

    async def stream(self, model: str, messages: Sequence[Message], **kwargs) -> AsyncGenerator[AssistantMessage, None]:
        if self.responses:
            yield self.responses.pop(0)

    async def close(self): pass


class TestLLMProvider(LLMProvider):
    def __init__(self, name: str, responses: list[str] = None, fail_on_stream: bool = False):
        self.name = name
        self.responses = responses or ["hello", "world"]
        self.fail_on_stream = fail_on_stream
        self.stream_calls: list[dict] = []
        self.close_called = False

    async def count_tokens(self, model: str, messages: Sequence[Message], **kwargs) -> int:
        return len(self.responses)

    async def send(self, model: str, messages: Sequence[Message], tools: Sequence[Tool], **kwargs) -> AssistantMessage:
        pass

    async def stream(self, model: str, messages: Sequence[Message], **kwargs) -> AsyncGenerator[AssistantMessage, None]:
        # Store calls for verification
        self.stream_calls.append({
            "model": model,
            "messages": list(messages),
            "kwargs": kwargs
        })

        if self.fail_on_stream:
            raise RuntimeError(f"Simulated failure in {self.name}")

        for content in self.responses:
            yield AssistantMessage(role="assistant",
                                   content=[Text(type="text", text=content)],
                                   id="test-id",
                                   parent_id="parent-id",
                                   is_aggregate=True)
            await asyncio.sleep(0.001)

    async def close(self):
        self.close_called = True


# A middleware that we can configure to do different things
class SpyMiddleware:
    def __init__(self, tag: str, mutate_model_to: str = None):
        self.tag = tag
        self.mutate_model_to = mutate_model_to
        self.call_log = []

    async def __call__(self, handler: StreamIterationHandler, ctx: StreamIterationContext, **kwargs) \
            -> AsyncGenerator[AssistantMessage, None]:
        self.call_log.append("enter")

        # Scenario: Mutating the context
        if self.mutate_model_to:
            ctx.model = self.mutate_model_to

        # Pass through
        async for msg in handler(ctx, **kwargs):
            yield msg

        self.call_log.append("exit")
