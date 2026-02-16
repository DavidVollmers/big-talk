import asyncio
import time

import pytest

from big_talk import AssistantMessage, ToolUse, Text
from tests.helpers import MockToolProvider


@pytest.mark.asyncio
async def test_tool_execution_persistence(bigtalk, simple_message):
    """
    Verify that tool results are yielded back to the user
    so they can update their history.
    """

    # 1. Define Tool
    async def my_tool():
        return "Tool Output"

    # 2. Setup Provider
    # Turn 1: LLM calls tool
    tool_call_msg = AssistantMessage(
        role="assistant",
        content=[ToolUse(type="tool_use", id="call_1", name="my_tool", params={})],
        id="msg_1", parent_id="p_1", is_aggregate=True
    )
    # Turn 2: LLM replies (after seeing tool result)
    final_reply_msg = AssistantMessage(
        role="assistant",
        content=[Text(type="text", text="I ran the tool.")],
        id="msg_2", parent_id="p_2", is_aggregate=True
    )

    provider = MockToolProvider([tool_call_msg, final_reply_msg])
    bigtalk.add_provider("test", lambda: provider)

    # 3. The "App" Logic
    history = [simple_message]  # Start with User message

    # We simulate a real app loop here
    async for msg in bigtalk.stream("test/model", history, tools=[my_tool]):
        # We only append AGGREGATE messages to our history
        # (BigTalk yields deltas too, but our mock yields full messages)
        if msg.get('is_aggregate') or msg['role'] == 'system':
            history.append(msg)

    # 4. Assertions on the UPDATED history
    assert len(history) == 4

    # Index 0: User ("Hello")
    assert history[0]['role'] == 'user'

    # Index 1: Assistant (Tool Call)
    assert history[1]['role'] == 'assistant'
    assert history[1]['content'][0]['type'] == 'tool_use'

    # Index 2: System (Tool Result) - THIS was missing before!
    assert history[2]['role'] == 'system'
    assert history[2]['content'][0]['result'] == "Tool Output"

    # Index 3: Assistant (Final Reply)
    assert history[3]['role'] == 'assistant'
    assert history[3]['content'][0]['text'] == "I ran the tool."


@pytest.mark.asyncio
async def test_parallel_execution_speed(bigtalk, simple_message):
    """Verify tools run in parallel."""

    async def slow_1():
        await asyncio.sleep(0.1)
        return "1"

    async def slow_2():
        await asyncio.sleep(0.1)
        return "2"

    tool_msg = AssistantMessage(
        role="assistant",
        content=[
            ToolUse(type="tool_use", id="a", name="slow_1", params={}),
            ToolUse(type="tool_use", id="b", name="slow_2", params={})
        ],
        id="m1", parent_id="p", is_aggregate=True
    )

    # Note: We only need 1 response from provider,
    # then the loop finishes because there are no more tool calls
    bigtalk.add_provider("test", lambda: MockToolProvider([tool_msg]))

    history = [simple_message]

    start = time.time()
    async for msg in bigtalk.stream("test/model", history, tools=[slow_1, slow_2]):
        pass
    duration = time.time() - start

    assert duration < 0.18  # Parallel (0.1s) vs Serial (0.2s)
