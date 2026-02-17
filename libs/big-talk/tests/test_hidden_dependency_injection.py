import inspect

import pytest
from big_talk import AssistantMessage, ToolUse
from big_talk.tool import tool


# 1. Define the Dependency Marker
class Dependency:
    """Marker class for dependencies."""
    pass


@pytest.mark.asyncio
async def test_hidden_dependency_injection(bigtalk, simple_message):
    """
    Verify that parameters marked as hidden:
    1. Do NOT appear in the Tool Schema (LLM doesn't see them).
    2. CAN be injected by middleware at runtime.
    """

    # --- A. Setup ---

    # A fake client we want to inject
    class DatabaseClient:
        def query(self): return "DB_RESULT"

    # The Tool definition
    @tool(hidden_default_types=(Dependency,))
    async def query_db(query_str: str, db: DatabaseClient = Dependency()):
        """Query the database."""
        return f"{query_str} -> {db.query()}"

    # --- B. Verification 1: Schema Hiding ---

    # The 'db' parameter should be missing from the JSON schema
    params = query_db.parameters['properties']
    assert 'query_str' in params
    assert 'db' not in params
    assert 'db' not in query_db.parameters['required']

    # --- C. Verification 2: Runtime Injection ---

    # Middleware to inject the dependency
    @bigtalk.tool_execution.use
    async def inject_db(handler, ctx, **kwargs):
        # Scan for Dependency params and inject the real client
        for tool_use in ctx.tool_uses:
            tool_name = tool_use['name']
            tool_def = next(t for t in ctx.tools if t.name == tool_name)

            sig = inspect.signature(tool_def.func)
            for name, param in sig.parameters.items():
                if isinstance(param.default, Dependency):
                    # INJECTION HAPPENS HERE
                    tool_use['params'][name] = DatabaseClient()

        return await handler(ctx, **kwargs)

    # Mock an LLM calling the tool (LLM only sends 'query_str')
    tool_msg = AssistantMessage(
        role="assistant",
        content=[ToolUse(type="tool_use", id="1", name="query_db", params={"query_str": "SELECT *"})],
        id="m1", parent_id="p1", is_aggregate=True
    )

    from tests.helpers import MockToolProvider
    bigtalk.add_provider("test", lambda: MockToolProvider([tool_msg]))

    # Execute
    history = []
    async for msg in bigtalk.stream("test/model", [simple_message], tools=[query_db]):
        if msg['role'] == 'tool': history.append(msg)

    # Assert Result
    result = history[0]['content'][0]['result']
    assert result == "SELECT * -> DB_RESULT"
