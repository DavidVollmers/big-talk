import pytest
from big_talk.tool import tool


def test_tool_docstring_formatting_kwargs():
    """Test @tool(variable='value')"""

    @tool(name="World")
    def hello():
        """Say Hello to {name}."""
        pass

    assert "Say Hello to World." in hello.description


def test_tool_docstring_formatting_args():
    """Test @tool('Value')"""

    @tool("BigTalk")
    def guide():
        """Welcome to {}."""
        pass

    assert "Welcome to BigTalk." in guide.description


def test_tool_brace_safety():
    """Test that JSON in docstrings doesn't crash the formatter."""

    # This would normally crash .format() because of {"foo": "bar"}
    @tool(formatting_var="Ignored")
    def json_tool():
        """Returns a dict like {"foo": "bar"}."""
        pass

    # It should fall back to the raw string instead of crashing
    assert '{"foo": "bar"}' in json_tool.description
