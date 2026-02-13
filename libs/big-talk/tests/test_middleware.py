from typing import Any, AsyncGenerator

import pytest

from big_talk import Message, BigTalk
from big_talk.streaming import StreamHandler, StreamContext
from tests.helpers import TestLLMProvider


@pytest.mark.asyncio
async def test_middleware_can_route_to_different_provider():
    bt = BigTalk()

    # 1. Setup two providers
    provider_a = TestLLMProvider("A")
    provider_b = TestLLMProvider("B")

    bt.add_provider("provA", lambda: provider_a)
    bt.add_provider("provB", lambda: provider_b)

    # 2. Add middleware that switches A -> B
    async def switching_middleware(handler: StreamHandler, ctx: StreamContext, **kwargs: Any) -> AsyncGenerator[
        Message, None]:
        if ctx.model == "provA/model-a":
            # Switch the model dynamically!
            ctx.model = "provB/model-b"
        async for msg in handler(ctx, **kwargs):
            yield msg

    # Manually wrapping simpler function into class if needed, or using your wrapper logic
    # For testing, you might need a helper to adapt simple funcs to StreamMiddleware protocol
    # ... assuming you added that adaptation or use a class:
    bt.add_middleware(switching_middleware)

    # 3. Call with Provider A
    messages = [Message(role="user", content="hi")]
    response = bt.stream("provA/model-a", messages)

    # 4. Consume
    async for _ in response: pass

    # 5. VERIFY
    # Provider A should NOT be touched
    assert len(provider_a.stream_calls) == 0

    # Provider B SHOULD be called with the new model name
    assert len(provider_b.stream_calls) == 1
    assert provider_b.stream_calls[0]['model'] == "model-b"
