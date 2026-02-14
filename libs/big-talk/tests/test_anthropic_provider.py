import pytest
from unittest.mock import MagicMock

from big_talk import SystemMessage, UserMessage
from big_talk.llm.anthropic_provider import AnthropicProvider
from big_talk.message import Message, ToolResult


# Mock the Anthropic stream events
class MockStream:
    def __init__(self, events):
        self.events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)


@pytest.fixture
def anthropic_provider():
    # Patch the import inside the class or mock the client directly
    provider = AnthropicProvider()
    provider._client = MagicMock()
    return provider


@pytest.mark.asyncio
async def test_anthropic_message_conversion(anthropic_provider):
    """Test that System Tool Results are moved to User role."""
    messages = [
        SystemMessage(role="system", content=[
            ToolResult(type="tool_result", tool_use_id="123", result="Success", is_error=False)
        ]),
        UserMessage(role="user", content="Hello", id="u1")
    ]

    system, converted, _ = anthropic_provider._convert_messages(messages)

    # Verify System message became a User message with tool_result
    assert converted[0]["role"] == "user"
    assert converted[0]["content"][0]["type"] == "tool_result"
    assert converted[0]["content"][0]["tool_use_id"] == "123"


@pytest.mark.asyncio
async def test_anthropic_streaming(anthropic_provider):
    """Test deltas vs aggregate messages."""

    # Mock events coming from Anthropic
    mock_events = [
        # 1. Text Delta
        MagicMock(type='content_block_stop', content_block=MagicMock(type='text', text="Hello")),
        # 2. Tool Use Delta
        MagicMock(type='content_block_stop',
                  content_block=MagicMock(type='tool_use', id='t1', name='search', input={})),
        # 3. Message Stop (End of Stream)
        MagicMock(type='message_stop')
    ]

    anthropic_provider._client.messages.stream.return_value = MockStream(mock_events)

    stream = anthropic_provider.stream("claude-3", [UserMessage(role="user", content="Hi", id="u1")])

    results = [msg async for msg in stream]

    # Expect: 2 Deltas + 1 Aggregate
    assert len(results) == 3

    # Check Delta 1
    assert results[0]["is_aggregate"] is False
    assert results[0]["content"][0]["text"] == "Hello"

    # Check Delta 2
    assert results[1]["is_aggregate"] is False
    assert results[1]["content"][0]["type"] == "tool_use"

    # Check Aggregate
    assert results[2]["is_aggregate"] is True
    assert len(results[2]["content"]) == 2  # Should contain both text and tool
