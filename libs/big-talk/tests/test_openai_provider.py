import pytest
from unittest.mock import AsyncMock, MagicMock

from big_talk import UserMessage
from big_talk.llm.openai_provider import OpenAIProvider
from big_talk.message import Message
from big_talk.tool import Tool


# Helper to create OpenAI-style chunks
def create_chunk(text=None, tool_calls=None):
    delta = MagicMock()
    delta.content = text
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


# Helper to create tool call chunks
def create_tool_chunk(index, id=None, name=None, args=None):
    tc = MagicMock()
    tc.index = index
    tc.id = id
    tc.function.name = name
    tc.function.arguments = args
    return [tc]


@pytest.fixture
def openai_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")
    provider = OpenAIProvider()
    provider._client = MagicMock()
    provider._client.chat.completions.create = AsyncMock()
    return provider


@pytest.mark.asyncio
async def test_openai_reactive_streaming(openai_provider):
    """Test that Text is yielded BEFORE tools start, and Tools yielded on index switch."""

    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = [
        # 1. Stream Text
        create_chunk(text="Checking "),
        create_chunk(text="weather..."),

        # 2. Start Tool 0 (Should trigger Text yield)
        create_chunk(tool_calls=create_tool_chunk(0, id="call_1", name="get_weather", args='{"loc":')),
        create_chunk(tool_calls=create_tool_chunk(0, args='"NYC"}')),

        # 3. Start Tool 1 (Should trigger Tool 0 yield)
        create_chunk(tool_calls=create_tool_chunk(1, id="call_2", name="get_time", args='{}')),
    ]

    openai_provider._client.chat.completions.create.return_value = mock_stream

    stream = openai_provider.stream("gpt-4", [UserMessage(role="user", content="Hi", id="u1")])
    results = [msg async for msg in stream]

    # Expect:
    # 1. Text Block (Checking weather...) - yielded when Tool 0 started
    # 2. Tool 0 Block (get_weather) - yielded when Tool 1 started
    # 3. Tool 1 Block (get_time) - yielded at stream end
    # 4. Aggregate Message - yielded at stream end

    assert len(results) == 4

    # Verify Text Block
    assert results[0]["content"][0]["text"] == "Checking weather..."
    assert results[0]["is_aggregate"] is False

    # Verify Tool 0
    assert results[1]["content"][0]["name"] == "get_weather"
    assert results[1]["content"][0]["params"] == {"loc": "NYC"}

    # Verify Tool 1
    assert results[2]["content"][0]["name"] == "get_time"


def test_openai_token_counting_with_tools(openai_provider):
    """Test the complex logic for counting tool definition tokens."""

    # Mock tiktoken
    mock_encoding = MagicMock()
    mock_encoding.encode = lambda s: [1] * len(s)  # Mock: 1 char = 1 token
    openai_provider._tiktoken = MagicMock()
    openai_provider._tiktoken.encoding_for_model.return_value = mock_encoding

    def my_tool(x: int):
        """Desc."""
        pass

    tools = [Tool.from_func(my_tool)]
    messages = [UserMessage(role="user", content="hi", id="u1")]

    # We just want to ensure it runs without error and returns a number > 0
    # since exact count depends on the mock encoding logic.
    # Note: You defined count_tokens as async in your provider!
    import asyncio
    count = asyncio.run(openai_provider.count_tokens("gpt-4", messages, tools=tools))

    assert count > 0
