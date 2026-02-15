from unittest.mock import MagicMock

import pytest

from big_talk import BigTalk


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
    assert results[0]['content'][0]['text'] == "hello"

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


def test_add_provider_duplicate_error():
    """Ensure adding a duplicate provider raises ValueError by default."""
    bt = BigTalk()
    mock_factory = lambda: MagicMock()

    # 1. Add first time
    bt.add_provider("test_provider", mock_factory)

    # 2. Add second time (should fail)
    with pytest.raises(ValueError, match='Provider "test_provider" is already registered'):
        bt.add_provider("test_provider", mock_factory)


def test_add_provider_override_success():
    """Ensure override=True replaces the provider and clears the cache."""
    bt = BigTalk()

    # Setup two different mocks to distinguish them
    factory_1 = MagicMock(return_value="Provider 1")
    factory_2 = MagicMock(return_value="Provider 2")

    # 1. Register first provider
    bt.add_provider("my_llm", factory_1)

    # 2. Instantiate it (to populate _providers cache)
    # We access the private method or just use get_llm_provider logic if exposed
    # For testing, we can simulate the cache population:
    bt._providers["my_llm"] = factory_1()
    assert bt._providers["my_llm"] == "Provider 1"

    # 3. Override with second provider
    bt.add_provider("my_llm", factory_2, override=True)

    # 4. Assertions
    # Factory should be updated
    assert bt._provider_factories["my_llm"] == factory_2

    # Cache should be CLEARED (critical for correct lifecycle management)
    assert "my_llm" not in bt._providers

    # 5. Verify the new provider is used on next access
    # (Assuming we have a way to access it, or just checking internal state)
    assert bt._provider_factories["my_llm"] is factory_2


def test_override_default_provider_with_kwargs():
    """Verify a user can override the default 'anthropic' provider with a configured one."""
    bt = BigTalk()

    # User wants to set a specific timeout
    def configured_anthropic_factory():
        # This requires the real package or a mock,
        # so for unit tests we usually mock the Provider class itself
        mock_provider = MagicMock()
        mock_provider.name = "configured"
        return mock_provider

    bt.add_provider("anthropic", configured_anthropic_factory, override=True)

    # Ensure the override worked
    assert bt._provider_factories["anthropic"] == configured_anthropic_factory
