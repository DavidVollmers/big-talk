import pytest
from unittest.mock import MagicMock, AsyncMock
from big_talk import BigTalk, Tool


@pytest.mark.asyncio
async def test_stream_normalizes_tools(simple_message):
    """Ensure raw functions passed to stream() are converted to Tools."""

    def my_raw_func(a: int):
        """My raw doc."""
        pass

    bt = BigTalk()
    mock_provider = MagicMock()

    async def mock_stream_gen(*args, **kwargs):
        if False:
            yield

    mock_provider.stream.side_effect = mock_stream_gen

    bt.add_provider("mock", lambda: mock_provider)

    # Call with raw function
    async for _ in bt.stream("mock/model", [simple_message], tools=[my_raw_func]):
        pass

    # Verify provider received a Tool object, not the function
    call_args = mock_provider.stream.call_args
    passed_tools = call_args.kwargs["tools"]

    assert len(passed_tools) == 1
    assert isinstance(passed_tools[0], Tool)
    assert passed_tools[0].name == "my_raw_func"
