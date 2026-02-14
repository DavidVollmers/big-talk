import pytest


@pytest.mark.asyncio
async def test_happy_path_routing(bigtalk, create_provider, simple_message):
    provider = create_provider(name="test-4")
    bigtalk.add_provider("test", lambda: provider)

    # Action
    response = bigtalk.stream("test/test-4", [simple_message])

    # Consume
    results = [msg async for msg in response]

    # Assert
    assert len(results) == 2
    assert results[0]['content'] == "hello"

    # Verify the provider received the stripped model name
    assert len(provider.stream_calls) == 1
    assert provider.stream_calls[0]["model"] == "test-4"


@pytest.mark.asyncio
async def test_lazy_loading(bigtalk, create_provider, simple_message):
    """Verify factory is NOT called until stream() is called."""
    factory_called = False

    def factory():
        nonlocal factory_called
        factory_called = True
        return create_provider()

    bigtalk.add_provider("lazy", factory)

    assert not factory_called

    # Trigger load
    async for _ in bigtalk.stream("lazy/model", [simple_message]):
        pass

    assert factory_called


@pytest.mark.asyncio
async def test_singleton_reuse(bigtalk, create_provider, simple_message):
    """Verify we don't recreate the provider on second call."""
    call_count = 0

    def factory():
        nonlocal call_count
        call_count += 1
        return create_provider()

    bigtalk.add_provider("reuse", factory)

    async for _ in bigtalk.stream("reuse/m1", [simple_message]):
        pass
    async for _ in bigtalk.stream("reuse/m2", [simple_message]):
        pass

    assert call_count == 1  # Factory only called once


@pytest.mark.asyncio
async def test_invalid_inputs(bigtalk, simple_message):
    with pytest.raises(ValueError, match="Expected format"):
        async for _ in bigtalk.stream("bad-format", [{'role': 'user', 'content': 'test'}]):
            pass

    with pytest.raises(NotImplementedError, match="not supported"):
        async for _ in bigtalk.stream("unknown/model", [simple_message]):
            pass


@pytest.mark.asyncio
async def test_provider_stream_failure(bigtalk, create_provider, simple_message):
    """Ensure exceptions in the LLM bubble up to the user."""
    provider = create_provider(fail_on_stream=True)
    bigtalk.add_provider("fail", lambda: provider)

    with pytest.raises(RuntimeError, match="Simulated failure"):
        async for _ in bigtalk.stream("fail/m", [simple_message]):
            pass


@pytest.mark.asyncio
async def test_global_close(bigtalk, create_provider, simple_message):
    p1 = create_provider()
    p2 = create_provider()

    bigtalk.add_provider("p1", lambda: p1)
    bigtalk.add_provider("p2", lambda: p2)

    # Must initialize them first
    async for _ in bigtalk.stream("p1/m", [simple_message]):
        pass
    async for _ in bigtalk.stream("p2/m", [simple_message]):
        pass

    await bigtalk.close()

    assert p1.close_called
    assert p2.close_called
