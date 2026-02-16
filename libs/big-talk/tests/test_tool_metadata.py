import pytest
import asyncio
from typing import Any, Callable

from big_talk import BigTalk, AssistantMessage, ToolUse
from big_talk.tool import tool, Tool
from tests.helpers import MockToolProvider


# --- Fixtures ---

@pytest.fixture
def mock_tool_provider(bigtalk):
    def _create(responses):
        provider = MockToolProvider(responses)
        bigtalk.add_provider("test", lambda: provider)
        return provider

    return _create


# --- Tests ---

def test_tool_definition_metadata():
    """Unit test: Verify metadata is stored correctly on the Tool object."""

    @tool(metadata={"scope": "read", "cost": 10})
    def simple_tool(x: int):
        """A simple tool."""
        return x

    assert isinstance(simple_tool, Tool)
    assert simple_tool.metadata == {"scope": "read", "cost": 10}
    assert simple_tool.name == "simple_tool"


@pytest.mark.asyncio
async def test_execution_metadata_merging(bigtalk, mock_tool_provider, simple_message):
    """
    Verify that BaseToolExecutionHandler merges the static Tool metadata
    into the ToolUse object so that middleware can see it.
    """

    # 1. Define a tool with STATIC metadata
    @tool(metadata={"static_key": "static_value", "conflict": "static_wins?"})
    async def meta_tool(arg: str):
        return "done"

    # 2. Setup Middleware to SPY on the merging logic
    # The handler modifies tool_use in-place, so we can inspect it after handler() returns
    spy_data = {}

    @bigtalk.tool_execution.use
    async def spy_middleware(handler, ctx, **kwargs):
        # Run the handler (which creates tasks and merges metadata)
        tasks = await handler(ctx, **kwargs)

        # Capture the metadata from the first tool use
        spy_data['result'] = ctx.tool_uses[0].get('metadata')

        return tasks

    # 3. Setup LLM Response with RUNTIME metadata
    # (Simulating a scenario where we injected metadata earlier in the stream)
    tool_use = ToolUse(
        type="tool_use",
        id="1",
        name="meta_tool",
        params={"arg": "hi"},
        metadata={"runtime_key": "runtime_value", "conflict": "runtime_wins!"}
    )

    msg = AssistantMessage(
        role="assistant",
        content=[tool_use],
        id="m1", parent_id="p1", is_aggregate=True
    )

    mock_tool_provider([msg])

    # 4. Run Loop
    async for _ in bigtalk.stream("test/model", [simple_message], tools=[meta_tool]):
        pass

    # 5. Assertions
    merged = spy_data['result']

    # A. Static metadata should be present
    assert merged['static_key'] == "static_value"

    # B. Runtime metadata should be present
    assert merged['runtime_key'] == "runtime_value"

    # C. Runtime should OVERRIDE static on conflict
    # (Because we did {**tool.meta, **tool_use.meta})
    assert merged['conflict'] == "runtime_wins!"


@pytest.mark.asyncio
async def test_metadata_based_middleware_blocking(bigtalk, mock_tool_provider, simple_message):
    """
    Real-world scenario: Block execution if a tool is marked as 'dangerous'
    in its metadata.
    """

    # 1. Define Tools
    @tool(metadata={"safety": "safe"})
    async def safe_tool():
        return "Safe"

    @tool(metadata={"safety": "dangerous"})
    async def nuke_tool():
        return "Boom"

    # 2. Security Middleware
    @bigtalk.tool_execution.use
    async def security_guard(handler, ctx, **kwargs):
        # Step 1: Let the handler merge metadata
        tasks = await handler(ctx, **kwargs)

        # Step 2: Filter tasks based on the NOW MERGED metadata
        safe_tasks = []
        for i, tool_use in enumerate(ctx.tool_uses):
            safety_level = tool_use.get('metadata', {}).get('safety')

            if safety_level == 'dangerous':
                tasks[i].close()

                # Replace the real task with a blocked message
                async def blocked():
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use['id'],
                        "result": "Error: Tool execution blocked by security policy.",
                        "is_error": True
                    }

                safe_tasks.append(blocked())
            else:
                safe_tasks.append(tasks[i])

        return safe_tasks

    # 3. Trigger both tools
    msg = AssistantMessage(
        role="assistant",
        content=[
            ToolUse(type="tool_use", id="1", name="safe_tool", params={}),
            ToolUse(type="tool_use", id="2", name="nuke_tool", params={})
        ],
        id="m1", parent_id="p1", is_aggregate=True
    )
    mock_tool_provider([msg])

    # 4. Run
    history = []
    async for m in bigtalk.stream("test/model", [simple_message], tools=[safe_tool, nuke_tool]):
        if m['role'] == 'tool': history.append(m)

    # 5. Verify Results
    results = history[0]['content']

    res_1 = next(r for r in results if r['tool_use_id'] == '1')
    res_2 = next(r for r in results if r['tool_use_id'] == '2')

    assert res_1['result'] == "Safe"
    assert "blocked" in res_2['result']
    assert res_2['is_error'] is True
