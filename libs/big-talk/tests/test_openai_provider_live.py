"""Live tests against a real OpenAI-compatible endpoint.

Skipped unless BIGTALK_LIVE_OPENAI_BASE_URL / BIGTALK_LIVE_OPENAI_API_KEY are configured
(see .env.example). These make real network calls and are not part of CI.
"""
import os

import pytest
import pytest_asyncio

from big_talk import BigTalk, Tool, UserMessage
from big_talk.llm.openai import OpenAIProvider

BASE_URL = os.getenv("BIGTALK_LIVE_OPENAI_BASE_URL")
API_KEY = os.getenv("BIGTALK_LIVE_OPENAI_API_KEY")
MODEL_OVERRIDE = os.getenv("BIGTALK_LIVE_OPENAI_MODEL")

pytestmark = pytest.mark.skipif(
    not BASE_URL or not API_KEY,
    reason="live OpenAI-compatible endpoint not configured (see .env.example)",
)


@pytest_asyncio.fixture
async def live_provider():
    provider = OpenAIProvider(base_url=BASE_URL, api_key=API_KEY)
    yield provider
    await provider.close()


@pytest_asyncio.fixture
async def live_model(live_provider):
    if MODEL_OVERRIDE:
        return MODEL_OVERRIDE
    models = await live_provider._client.models.list()
    return models.data[0].id


@pytest.fixture
def live_bigtalk(live_provider):
    bt = BigTalk()
    bt.add_provider("openai", lambda: live_provider, override=True)
    return bt


@pytest.mark.asyncio
async def test_live_send(live_bigtalk, live_model):
    new_messages = await live_bigtalk.send(
        model=f"openai/{live_model}",
        messages=[UserMessage(role="user", content="Say the word 'pong' and nothing else.", id="u1")],
    )

    assistant_messages = [m for m in new_messages if m["role"] == "assistant"]
    assert assistant_messages

    response = assistant_messages[-1]
    text_blocks = [block for block in response["content"] if block["type"] == "text"]
    assert text_blocks
    assert text_blocks[0]["text"].strip()


@pytest.mark.asyncio
async def test_live_stream(live_bigtalk, live_model):
    aggregates = []
    async for message in live_bigtalk.stream(
        model=f"openai/{live_model}",
        messages=[UserMessage(role="user", content="Say the word 'pong' and nothing else.", id="u1")],
    ):
        if message.get("is_aggregate"):
            aggregates.append(message)

    assert aggregates
    text_blocks = [block for block in aggregates[-1]["content"] if block["type"] == "text"]
    assert text_blocks
    assert text_blocks[0]["text"].strip()


@pytest.mark.asyncio
async def test_live_tool_calling(live_bigtalk, live_model):
    """Honest probe: passes only if the hosted model/server supports OpenAI-style
    function calling. A failure here reflects the backend's capability, not a bug."""

    def get_weather(location: str) -> str:
        """Get the current weather for a location.

        :param location: The city to get the weather for.
        """
        return f"It is sunny in {location}."

    new_messages = await live_bigtalk.send(
        model=f"openai/{live_model}",
        messages=[UserMessage(role="user", content="What is the weather in Berlin? Use the get_weather tool.",
                              id="u1")],
        tools=[Tool.from_func(get_weather)],
        max_iterations=1,
    )

    assistant_messages = [m for m in new_messages if m["role"] == "assistant"]
    assert assistant_messages

    tool_uses = [block for block in assistant_messages[0]["content"] if block["type"] == "tool_use"]
    assert tool_uses
    assert tool_uses[0]["name"] == "get_weather"
