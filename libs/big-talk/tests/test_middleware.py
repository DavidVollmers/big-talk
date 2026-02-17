import pytest

from big_talk import AssistantMessage, Text, AppMessage, ToolUse


@pytest.mark.asyncio
async def test_middleware_execution_order(bigtalk, create_provider, simple_message):
    """Verify 'Onion' architecture: Outer -> Inner -> LLM -> Inner -> Outer"""
    call_log = []

    async def mw1(handler, ctx, **kwargs):
        call_log.append("mw1_enter")
        async for msg in handler(ctx, **kwargs):
            yield msg
        call_log.append("mw1_exit")

    async def mw2(handler, ctx, **kwargs):
        call_log.append("mw2_enter")
        async for msg in handler(ctx, **kwargs):
            yield msg
        call_log.append("mw2_exit")

    bigtalk.add_provider("test", lambda: create_provider())
    bigtalk.streaming.use(mw1)
    bigtalk.streaming.use(mw2)

    async for _ in bigtalk.stream("test/model", [simple_message]):
        pass

    assert call_log == ["mw1_enter", "mw2_enter", "mw2_exit", "mw1_exit"]


@pytest.mark.asyncio
async def test_middleware_context_mutation(bigtalk, create_provider, simple_message):
    """CRITICAL: Test that middleware can change the model and routing updates dynamically."""
    prov_a = create_provider(name="A")
    prov_b = create_provider(name="B")

    bigtalk.add_provider("provA", lambda: prov_a)
    bigtalk.add_provider("provB", lambda: prov_b)

    async def router_middleware(handler, ctx, **kwargs):
        if ctx.model == "provA/expensive":
            ctx.model = "provB/cheap"
        async for msg in handler(ctx, **kwargs):
            yield msg

    bigtalk.streaming.use(router_middleware)

    # Call with A
    async for _ in bigtalk.stream("provA/expensive", [simple_message]):
        pass

    # Assert A was IGNORED and B was USED
    assert len(prov_a.stream_calls) == 0
    assert len(prov_b.stream_calls) == 1
    assert prov_b.stream_calls[0]["model"] == "cheap"


@pytest.mark.asyncio
async def test_middleware_short_circuit(bigtalk, create_provider, simple_message):
    """Test middleware returning early (caching) without calling handler."""
    provider = create_provider()
    bigtalk.add_provider("test", lambda: provider)

    async def cache_middleware(handler, ctx, **kwargs):
        # Don't call handler, just yield mock response
        yield AssistantMessage(role="assistant", content=[Text(type="text", text="cached_response")], id="cached_id",
                               parent_id="cached_parent", is_aggregate=True)

    bigtalk.streaming.use(cache_middleware)

    results = [m async for m in bigtalk.stream("test/m", [simple_message])]

    assert len(results) == 1
    assert results[0]['content'][0]['text'] == "cached_response"
    assert len(provider.stream_calls) == 0  # LLM never touched


@pytest.mark.asyncio
async def test_middleware_app_message(bigtalk, create_provider, simple_message):
    """Test middleware returning early (caching) without calling handler."""
    provider = create_provider()
    bigtalk.add_provider("test", lambda: provider)

    async def app_middleware(handler, ctx, **kwargs):
        # Don't call handler, just yield mock response
        yield AppMessage(role="app", content=[Text(type="text", text="app_response")], id="cached_id", type="test")

    bigtalk.streaming.use(app_middleware)

    results = [m async for m in bigtalk.stream("test/m", [simple_message])]

    assert len(results) == 1
    assert results[0]['content'][0]['text'] == "app_response"
    assert len(provider.stream_calls) == 0  # LLM never touched


@pytest.mark.asyncio
async def test_middleware_argument_injection(bigtalk, create_provider, simple_message):
    provider = create_provider()
    bigtalk.add_provider("test", lambda: provider)

    async def inject_middleware(handler, ctx, **kwargs):
        kwargs["temperature"] = 0.99
        async for msg in handler(ctx, **kwargs):
            yield msg

    bigtalk.streaming.use(inject_middleware)

    async for _ in bigtalk.stream("test/m", [simple_message]):
        pass

    assert provider.stream_calls[0]["kwargs"]["temperature"] == 0.99


@pytest.mark.asyncio
async def test_middleware_can_resolve_provider_manually(bigtalk, create_provider, simple_message):
    """Test that middleware can ask 'who is the provider?' before streaming."""
    provider = create_provider()
    bigtalk.add_provider("test", lambda: provider)

    resolved_provider = None

    async def inspection_middleware(handler, ctx, **kwargs):
        nonlocal resolved_provider
        # This is the "Chicken and Egg" feature we built
        resolved_provider, _ = ctx.get_llm_provider()
        async for msg in handler(ctx, **kwargs):
            yield msg

    bigtalk.streaming.use(inspection_middleware)
    async for _ in bigtalk.stream("test/m", [simple_message]):
        pass

    assert resolved_provider is provider


@pytest.mark.asyncio
async def test_middleware_history_loading_deduplication(bigtalk, create_provider, simple_message):
    """
    Verify that middleware using 'ctx.iteration' only injects history ONCE.
    This prevents the "Infinite Duplication" bug during tool loops.
    """

    # 1. Setup Mock "Database" History
    # Assume we have 2 old messages in the DB
    db_history = [
        AssistantMessage(role="assistant", content=[Text(type="text", text="Old Msg 1")], id="old_1",
                         is_aggregate=True),
        AssistantMessage(role="assistant", content=[Text(type="text", text="Old Msg 2")], id="old_2", is_aggregate=True)
    ]

    # 2. Define the "History Middleware" (The Fix)
    history_loaded_count = 0

    async def history_middleware(handler, ctx, **kwargs):
        nonlocal history_loaded_count

        # THE FIX: Only load history if iteration is 0
        if ctx.iteration == 0:
            history_loaded_count += 1
            # Prepend DB history to the current input
            ctx.messages[:] = db_history + ctx.messages

        async for msg in handler(ctx, **kwargs):
            yield msg

    bigtalk.streaming.use(history_middleware)

    # 3. Setup Provider to Force a Loop (Tool Use)
    # Turn 1: Returns Tool Call
    # Turn 2: Returns Final Answer
    tool_use_msg = AssistantMessage(
        role="assistant",
        content=[ToolUse(type="tool_use", id="call_1", name="test_tool", params={})],
        id="msg_tool",
        is_aggregate=True
    )
    final_msg = AssistantMessage(
        role="assistant",
        content=[Text(type="text", text="Final Answer")],
        id="msg_final",
        is_aggregate=True
    )

    mock_provider = create_provider()

    captured_message_lists = []

    async def mock_stream_gen(model, messages, **kwargs):
        # Store a copy of the message list for inspection
        captured_message_lists.append(list(messages))

        # Logic to simulate tool conversation
        current_len = len(messages)

        # Iteration 0: Expecting 3 messages (2 DB + 1 Input)
        if current_len == 3:
            yield tool_use_msg
        # Iteration 1: Expecting 5 messages (2 DB + 1 Input + 1 ToolUse + 1 ToolResult)
        else:
            yield final_msg

    mock_provider.stream = mock_stream_gen
    bigtalk.add_provider("test", lambda: mock_provider)

    # 4. Mock Tool Execution
    # We need a dummy tool so the loop continues
    async def test_tool():
        return "Tool Result Data"

    # 5. Run Stream (Should trigger 2 iterations)
    # We pass ONLY the new message here
    results = [m async for m in bigtalk.stream("test/model", [simple_message], tools=[test_tool])]

    # --- ASSERTIONS ---

    # A. Verify History was only loaded ONCE
    assert history_loaded_count == 1, "Middleware should only load history on iteration 0"

    # B. Verify Result contains the final answer
    assert results[-1]['content'][0]['text'] == "Final Answer"

    # C. Verify NO Duplication in the Context passed to the Provider
    assert len(captured_message_lists) == 2, "Provider should have been called exactly twice (Iteration 0 and 1)"

    # Call 1 (Iteration 0): [Old1, Old2, UserInput]
    assert len(
        captured_message_lists[0]) == 3, f"Iteration 0 should have 3 messages, got {len(captured_message_lists[0])}"
    assert captured_message_lists[0][0]['id'] == "old_1"

    # Call 2 (Iteration 1): [Old1, Old2, UserInput, ToolUse, ToolResult]
    # If middleware duplicated history, this would be 3 + 2 (DB) + 2 (Tools) = 7 or more!
    assert len(captured_message_lists[
                   1]) == 5, f"Iteration 1 should have 5 messages, got {len(captured_message_lists[1])}. Duplication detected!"

    # Verify the sequence is clean
    ids = [m['id'] for m in captured_message_lists[1]]
    assert ids == ["old_1", "old_2", simple_message['id'], "msg_tool", ids[4]]  # ids[4] is generated tool result ID
