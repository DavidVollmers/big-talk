import asyncio

import pytest

from big_talk import tool


@pytest.mark.asyncio
async def test_manual_execution_success(bigtalk):
    """Verify simple tool execution works."""

    @tool
    async def adder(a: int, b: int) -> int:
        return a + b

    # Execute manually
    result = await bigtalk.execute_tool(adder, {"a": 5, "b": 3})

    assert result == "8"  # Results are always stringified


@pytest.mark.asyncio
async def test_manual_execution_failure(bigtalk):
    """Verify tool errors are raised as Python exceptions."""

    @tool
    async def crasher():
        raise ValueError("Boom!")

    # Should raise Exception, not return "Error: Boom!" string
    with pytest.raises(Exception) as excinfo:
        await bigtalk.execute_tool(crasher, {})

    assert "Boom!" in str(excinfo.value)


@pytest.mark.asyncio
async def test_manual_execution_middleware(bigtalk):
    """
    Critical Test: Verify that manual execution runs through the
    same middleware stack as the AI agent.
    """

    @tool
    async def echo(msg: str): return msg

    # 1. Add Middleware that intercepts execution
    # This proves we are using the unified stack
    intercepted_log = []

    @bigtalk.tool_execution.use
    async def logging_middleware(handler, ctx, **kwargs):
        # Log before execution
        intercepted_log.append(f"Calling: {ctx.tool_uses[0]['name']}")

        # Run
        tasks = await handler(ctx, **kwargs)

        # Log after (wait for result)
        results = await asyncio.gather(*tasks)
        intercepted_log.append(f"Result: {results[0]['result']}")

        # We must re-wrap the result into an awaitable because handler expects it
        async def wrap(r): return r

        return [wrap(r) for r in results]

    # 2. Execute Manually
    result = await bigtalk.execute_tool(echo, {"msg": "Hello World"})

    # 3. Verify Middleware Ran
    assert result == "Hello World"
    assert "Calling: echo" in intercepted_log
    assert "Result: Hello World" in intercepted_log


@pytest.mark.asyncio
async def test_manual_execution_metadata(bigtalk):
    """Verify metadata is passed correctly to the tool context."""

    @tool
    async def meta_tool(): return "ok"

    captured_meta = {}

    @bigtalk.tool_execution.use
    async def spy(handler, ctx, **kwargs):
        captured_meta.update(ctx.tool_uses[0].get('metadata', {}))
        return await handler(ctx, **kwargs)

    # Pass metadata during manual execution
    await bigtalk.execute_tool(
        meta_tool,
        {},
        metadata={"user_id": "123", "source": "api"}
    )

    assert captured_meta['user_id'] == "123"
    assert captured_meta['source'] == "api"
