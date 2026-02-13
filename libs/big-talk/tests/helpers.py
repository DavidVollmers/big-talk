import asyncio
from typing import AsyncGenerator, Iterable, Any
from big_talk.llm import LLMProvider
from big_talk import Message
from big_talk.streaming import StreamContext, StreamHandler


class TestLLMProvider(LLMProvider):
    def __init__(self, name: str, responses: list[str] = None, fail_on_stream: bool = False):
        self.name = name
        self.responses = responses or ["hello", "world"]
        self.fail_on_stream = fail_on_stream
        self.stream_calls = []  # To verify calls
        self.close_called = False

    async def count_tokens(self, model: str, messages: Iterable[Message], **kwargs) -> int:
        return len(self.responses)

    async def stream(self, model: str, messages: Iterable[Message], **kwargs) -> AsyncGenerator[Message, None]:
        self.stream_calls.append({"model": model, "messages": messages, "kwargs": kwargs})

        if self.fail_on_stream:
            raise RuntimeError(f"Simulated failure in {self.name}")

        for content in self.responses:
            yield Message(role="assistant", content=content)
            await asyncio.sleep(0.01)  # Simulate network lag

    async def close(self):
        self.close_called = True


# A middleware that we can configure to do different things
class SpyMiddleware:
    def __init__(self, tag: str, mutate_model_to: str = None):
        self.tag = tag
        self.mutate_model_to = mutate_model_to
        self.call_log = []

    async def __call__(self, handler: StreamHandler, ctx: StreamContext, **kwargs) -> AsyncGenerator[Message, None]:
        self.call_log.append("enter")

        # Scenario: Mutating the context
        if self.mutate_model_to:
            ctx.model = self.mutate_model_to

        # Pass through
        async for msg in handler(ctx, **kwargs):
            yield msg

        self.call_log.append("exit")
