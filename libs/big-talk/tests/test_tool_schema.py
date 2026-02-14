from typing import TypedDict, Annotated, Literal, List
from big_talk.tool import Tool


# --- Data Structures for Testing ---

class Address(TypedDict):
    street: str
    city: str


class UserProfile(TypedDict):
    age: Annotated[int, "Age in years"]
    status: Literal["active", "inactive"]
    address: Address


def simple_func(x: int, y: int) -> int:
    """Adds two numbers."""
    return x + y


def complex_func(profile: UserProfile, tags: List[str]):
    """
    Creates a user.

    Args:
        profile: The user profile data.
        tags: A list of tags to apply.
    """
    pass


# --- Tests ---

def test_simple_tool_schema():
    tool = Tool.from_func(simple_func)

    assert tool.name == "simple_func"
    assert tool.description == "Adds two numbers."
    assert tool.parameters["properties"]["x"]["type"] == "integer"
    assert tool.parameters["properties"]["y"]["type"] == "integer"
    assert "x" in tool.parameters["required"]


def test_complex_nested_schema():
    tool = Tool.from_func(complex_func)

    params = tool.parameters["properties"]

    # 1. Check TypedDict (profile)
    assert params["profile"]["type"] == "object"
    profile_props = params["profile"]["properties"]

    # 2. Check Annotated Description
    assert profile_props["age"]["type"] == "integer"
    # Note: Description might differ based on how you implemented combining docstring + annotated
    assert "Age in years" in profile_props["age"]["description"]

    # 3. Check Literal Enum
    assert profile_props["status"]["type"] == "string"
    assert profile_props["status"]["enum"] == ["active", "inactive"]

    # 4. Check Nested TypedDict (address)
    assert profile_props["address"]["type"] == "object"
    assert profile_props["address"]["properties"]["city"]["type"] == "string"

    # 5. Check List[str]
    assert params["tags"]["type"] == "array"
    assert params["tags"]["items"]["type"] == "string"

    # 6. Check Docstring Injection
    # "The user profile data." comes from docstring, "Age in years" comes from Annotated
    assert "The user profile data" in params["profile"]["description"]


def test_annotated_stripping_bug_fix():
    """Ensure Annotated types are not stripped and ignored."""

    def func(val: Annotated[int, "Must be positive"]): pass

    tool = Tool.from_func(func)
    # If this fails, you forgot include_extras=True in get_type_hints
    assert "Must be positive" in tool.parameters["properties"]["val"]["description"]
