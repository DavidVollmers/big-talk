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
        if msg.get('is_aggregate') or msg['role'] == 'tool':
            history.append(msg)

    # 4. Assertions on the UPDATED history
    assert len(history) == 4

    # Index 0: User ("Hello")
    assert history[0]['role'] == 'user'

    # Index 1: Assistant (Tool Call)
    assert history[1]['role'] == 'assistant'
    assert history[1]['content'][0]['type'] == 'tool_use'

    # Index 2: System (Tool Result) - THIS was missing before!
    assert history[2]['role'] == 'tool'
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


@pytest.mark.asyncio
async def test_tool_error_handling(bigtalk, simple_message):
    """
    Verify that if one tool fails, others still run, and the error
    is reported in the SystemMessage.
    """

    # 1. Define Tools
    async def success_tool():
        return "Success"

    async def fail_tool():
        raise ValueError("Boom!")

    # 2. Mock LLM Response: Call both
    tool_msg = AssistantMessage(
        role="assistant",
        content=[
            ToolUse(type="tool_use", id="1", name="success_tool", params={}),
            ToolUse(type="tool_use", id="2", name="fail_tool", params={})
        ],
        id="resp_1", parent_id="p_1", is_aggregate=True
    )

    bigtalk.add_provider("test", lambda: MockToolProvider([tool_msg]))

    # 3. Execution
    history = [simple_message]

    async for msg in bigtalk.stream("test/model", history, tools=[success_tool, fail_tool]):
        if msg['role'] == 'tool':
            history.append(msg)

    # 4. Verification
    # Get the system message with results
    system_msg = history[-1]
    assert system_msg['role'] == 'tool'
    results = system_msg['content']

    # Find results by ID
    res_success = next(r for r in results if r['tool_use_id'] == '1')
    res_fail = next(r for r in results if r['tool_use_id'] == '2')

    # Success Check
    assert res_success['is_error'] is False
    assert res_success['result'] == "Success"

    # Failure Check
    assert res_fail['is_error'] is True
    assert "Boom!" in res_fail['result']


@pytest.mark.asyncio
async def test_tool_middleware_interception(bigtalk, simple_message):
    """
    Verify middleware can intercept the execution pipeline and modify results.
    """

    async def echo_tool(val: str):
        return val

    # 1. Register Middleware
    @bigtalk.tool_execution.use
    async def interception_middleware(handler, ctx, **kwargs):
        # A. Inspect: We can see what tools are being called
        assert ctx.tool_uses[0]['name'] == 'echo_tool'

        # B. Get Pending Tasks
        tasks = await handler(ctx, **kwargs)

        # C. Modify: Wrap the task to append text to the result
        async def wrapper(coro):
            res = await coro
            res['result'] = f"Intercepted: {res['result']}"
            return res

        return [wrapper(t) for t in tasks]

    # 2. Mock LLM
    tool_msg = AssistantMessage(
        role="assistant",
        content=[ToolUse(type="tool_use", id="1", name="echo_tool", params={"val": "Hello"})],
        id="resp_1", parent_id="p_1", is_aggregate=True
    )
    bigtalk.add_provider("test", lambda: MockToolProvider([tool_msg]))

    # 3. Execution
    history = [simple_message]
    async for msg in bigtalk.stream("test/model", history, tools=[echo_tool]):
        if msg['role'] == 'tool':
            history.append(msg)

    # 4. Verify Result
    result_str = history[-1]['content'][0]['result']
    assert result_str == "Intercepted: Hello"


@pytest.mark.asyncio
async def test_tool_not_found(bigtalk, simple_message):
    """Verify behavior when LLM calls a non-existent tool."""

    # Mock LLM calls 'ghost_tool'
    tool_msg = AssistantMessage(
        role="assistant",
        content=[ToolUse(type="tool_use", id="1", name="ghost_tool", params={})],
        id="resp_1", parent_id="p_1", is_aggregate=True
    )

    bigtalk.add_provider("test", lambda: MockToolProvider([tool_msg]))

    history = [simple_message]
    # We pass NO tools
    async for msg in bigtalk.stream("test/model", history, tools=[]):
        if msg['role'] == 'tool':
            history.append(msg)

    result = history[-1]['content'][0]

    assert result['is_error'] is True
    assert "not found" in result['result']


@pytest.mark.asyncio
async def test_tool_result_parent_linking(bigtalk, simple_message):
    """
    Verify that the ToolMessage containing the results points back
    to the AssistantMessage that requested the tool via parent_id.
    """

    async def my_tool():
        return "done"

    # Setup: Assistant Message with specific ID
    target_parent_id = "assist_123"

    tool_msg = AssistantMessage(
        role="assistant",
        content=[ToolUse(type="tool_use", id="call_1", name="my_tool", params={})],
        id=target_parent_id,  # <--- The ID we expect to see in the result
        parent_id="p_1",
        is_aggregate=True
    )

    bigtalk.add_provider("test", lambda: MockToolProvider([tool_msg]))

    history = [simple_message]

    async for msg in bigtalk.stream("test/model", history, tools=[my_tool]):
        if msg['role'] == 'tool':
            history.append(msg)

    # Check the ToolMessage
    tool_result_msg = history[-1]
    assert tool_result_msg['role'] == 'tool'

    # The Critical Assertion
    assert tool_result_msg['parent_id'] == target_parent_id


@pytest.mark.asyncio
async def test_batch_tool_execution_grouping(bigtalk, simple_message):
    """
    Verify that if the stream yields multiple assistant messages with tool calls,
    BigTalk executes them all in parallel but groups the results into
    separate ToolMessages keyed by their specific parent_id.
    """

    async def tool_1():
        return "1"

    async def tool_2():
        return "2"

    # Message 1 calls tool_1
    msg_1 = AssistantMessage(
        role="assistant",
        content=[ToolUse(type="tool_use", id="c1", name="tool_1", params={})],
        id="parent_A",
        parent_id="p",
        is_aggregate=True
    )

    # Message 2 calls tool_2 (simulating a provider yielding multiple thoughts/steps)
    msg_2 = AssistantMessage(
        role="assistant",
        content=[ToolUse(type="tool_use", id="c2", name="tool_2", params={})],
        id="parent_B",
        parent_id="p",
        is_aggregate=True
    )

    # Provider yields both in one go
    bigtalk.add_provider("test", lambda: MockToolProvider([msg_1, msg_2]))

    tool_messages = []

    async for msg in bigtalk.stream("test/model", [simple_message], tools=[tool_1, tool_2]):
        if msg['role'] == 'tool':
            tool_messages.append(msg)

    # Should have 2 separate tool result messages (one for A, one for B)
    assert len(tool_messages) == 2

    # Find results by parent_id
    res_A = next((m for m in tool_messages if m['parent_id'] == 'parent_A'), None)
    res_B = next((m for m in tool_messages if m['parent_id'] == 'parent_B'), None)

    assert res_A is not None, "Missing result for parent_A"
    assert res_B is not None, "Missing result for parent_B"

    assert res_A['content'][0]['result'] == "1"
    assert res_B['content'][0]['result'] == "2"

