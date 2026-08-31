import pytest
from dotenv import load_dotenv

from big_talk import BigTalk, Message, UserMessage
from tests.helpers import TestLLMProvider

load_dotenv()


@pytest.fixture
def bigtalk():
    """Returns a fresh BigTalk instance for every test."""
    return BigTalk()


@pytest.fixture
def create_provider():
    """Factory fixture to create providers easily."""

    def _create(name: str = "test", responses=None, **kwargs):
        return TestLLMProvider(name, responses, **kwargs)

    return _create


@pytest.fixture
def simple_message():
    return UserMessage(role="user", content="Hello", id="msg-1")
