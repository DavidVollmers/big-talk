import pytest

from big_talk import AssistantMessage


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
        # noinspection PyArgumentList
        yield AssistantMessage(role="assistant", content="cached_response", id="cached_id")

    bigtalk.streaming.use(cache_middleware)

    results = [m async for m in bigtalk.stream("test/m", [simple_message])]

    assert len(results) == 1
    assert results[0]['content'] == "cached_response"
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
