import pytest


@pytest.mark.asyncio
async def test_happy_path_routing(bigtalk, create_provider, simple_message):
    provider = create_provider(name="gpt-4")
    bigtalk.add_provider("openai", lambda: provider)

    # Action
    response = bigtalk.stream("openai/gpt-4", [simple_message])

    # Consume
    results = [msg async for msg in response]

    # Assert
    assert len(results) == 2
    assert results[0]['content'] == "hello"

    # Verify the provider received the stripped model name
    assert len(provider.stream_calls) == 1
    assert provider.stream_calls[0]["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_lazy_loading(bigtalk, create_provider):
    """Verify factory is NOT called until stream() is called."""
    factory_called = False

    def factory():
        nonlocal factory_called
        factory_called = True
        return create_provider()

    bigtalk.add_provider("lazy", factory)

    assert not factory_called

    # Trigger load
    async for _ in bigtalk.stream("lazy/model", []):
        pass

    assert factory_called


@pytest.mark.asyncio
async def test_singleton_reuse(bigtalk, create_provider):
    """Verify we don't recreate the provider on second call."""
    call_count = 0

    def factory():
        nonlocal call_count
        call_count += 1
        return create_provider()

    bigtalk.add_provider("reuse", factory)

    async for _ in bigtalk.stream("reuse/m1", []):
        pass
    async for _ in bigtalk.stream("reuse/m2", []):
        pass

    assert call_count == 1  # Factory only called once


@pytest.mark.asyncio
async def test_invalid_inputs(bigtalk):
    with pytest.raises(ValueError, match="Expected format"):
        async for _ in bigtalk.stream("bad-format", []):
            pass

    with pytest.raises(NotImplementedError, match="not supported"):
        async for _ in bigtalk.stream("unknown/model", []):
            pass


@pytest.mark.asyncio
async def test_provider_stream_failure(bigtalk, create_provider):
    """Ensure exceptions in the LLM bubble up to the user."""
    provider = create_provider(fail_on_stream=True)
    bigtalk.add_provider("fail", lambda: provider)

    with pytest.raises(RuntimeError, match="Simulated failure"):
        async for _ in bigtalk.stream("fail/m", []):
            pass


@pytest.mark.asyncio
async def test_global_close(bigtalk, create_provider):
    p1 = create_provider()
    p2 = create_provider()

    bigtalk.add_provider("p1", lambda: p1)
    bigtalk.add_provider("p2", lambda: p2)

    # Must initialize them first
    async for _ in bigtalk.stream("p1/m", []):
        pass
    async for _ in bigtalk.stream("p2/m", []):
        pass

    await bigtalk.close()

    assert p1.close_called
    assert p2.close_called
