from typing import TypedDict, Annotated, Literal, List

from pydantic import BaseModel, Field

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


def test_pydantic_model_support():
    class User(BaseModel):
        id: int = Field(..., description="User ID")
        name: str = Field(..., description="Full Name")

    def create_user(user: User):
        """Creates a new user."""
        pass

    tool = Tool.from_func(create_user)

    props = tool.parameters["properties"]

    # Check if Pydantic schema was embedded correctly
    assert props["user"]["type"] == "object"
    assert props["user"]["properties"]["id"]["type"] == "integer"
    assert "User ID" in props["user"]["properties"]["id"]["description"]
    assert "id" in props["user"]["required"]


def test_annotated_union_unwrapping():
    """
    Edge Case 1: 'Stale Origin' Bug.
    Verify that Annotated[str | None] is treated as Optional[str],
    not as a raw Annotated object (which would crash schema gen).
    """

    # This uses Python 3.10+ pipe syntax inside Annotated
    def func(
            # The annotated wrapper makes get_origin return Annotated first.
            # We must peel it off, then realize inside is a UnionType (str | None).
            arg: Annotated[str | None, "A description"] = None
    ):
        pass

    tool = Tool.from_func(func)
    props = tool.parameters["properties"]

    # 1. Check it didn't crash
    assert "arg" in props

    # 2. Check it correctly identified 'str' as the core type
    # (If origin wasn't refreshed, this might default to object or crash)
    assert props["arg"]["type"] == "string"

    # 3. Check description persisted
    assert "A description" in props["arg"]["description"]

    # 4. Check it is NOT required (because of = None)
    assert "arg" not in tool.parameters["required"]


def test_typeddict_optional_pipe_syntax():
    """
    Edge Case 2: 'TypedDict Required' Bug.
    Verify that fields marked as `Type | None` in a TypedDict
    are NOT added to the 'required' list.
    """

    class Config(TypedDict):
        mandatory: int
        # Python 3.10+ syntax.
        # Previously, is_optional check failed for types.UnionType
        optional_pipe: str | None

    def configure(cfg: Config): pass

    tool = Tool.from_func(configure)
    schema = tool.parameters["properties"]["cfg"]
    required_fields = schema["required"]

    # 'mandatory' should be there
    assert "mandatory" in required_fields

    # 'optional_pipe' should be MISSING because it includes None
    assert "optional_pipe" not in required_fields

    # Verify types
    assert schema["properties"]["mandatory"]["type"] == "integer"
    assert schema["properties"]["optional_pipe"]["type"] == "string"


def test_modern_union_syntax():
    """
    Edge Case 3: General types.UnionType support.
    Verify that 'int | str' and 'str | None' work in standard args.
    """

    def processor(
            # Simple Optional
            filter_val: str | None = None,
            # Union of primitives (should default to first type or Any)
            id_val: int | str = 1
    ):
        pass

    tool = Tool.from_func(processor)
    props = tool.parameters["properties"]

    # 1. Optional[str] -> string
    assert props["filter_val"]["type"] == "string"
    assert "filter_val" not in tool.parameters["required"]

    # 2. Union[int, str] -> integer (Our logic picks the first one)
    assert props["id_val"]["type"] == "integer"


def test_pydantic_nested_definitions_hoisting():
    """
    Verify that nested Pydantic models have their $defs extracted
    and moved to the root of the tool schema.
    """

    class Address(BaseModel):
        street: str
        zip: str

    class User(BaseModel):
        name: str
        address: Address  # Nested Pydantic model

    def create_user(user: User):
        pass

    tool = Tool.from_func(create_user)

    # 1. Check Root Structure
    assert "$defs" in tool.parameters
    assert "Address" in tool.parameters["$defs"]

    # 2. Check Reference
    user_props = tool.parameters["properties"]["user"]["properties"]
    assert "$ref" in user_props["address"]
    assert user_props["address"]["$ref"] == "#/$defs/Address"


def test_pydantic_recursive_model():
    """
    Verify that recursive Pydantic models (referencing themselves)
    generate valid schemas with hoisted definitions.
    """

    class Category(BaseModel):
        name: str
        # Recursive reference
        subcategories: List["Category"] = Field(default_factory=list)

    def organize(root: Category):
        pass

    tool = Tool.from_func(organize)

    # 1. Definition exists
    assert "$defs" in tool.parameters
    assert "Category" in tool.parameters["$defs"]

    # 2. Schema points to itself correctly
    cat_def = tool.parameters["$defs"]["Category"]
    sub_items = cat_def["properties"]["subcategories"]["items"]
    assert sub_items["$ref"] == "#/$defs/Category"


def test_boss_level_schema():
    """
    Combines: TypedDict + Annotated + Optional (Pipe Syntax) + Description Injection.
    Ensures that the strictness logic works even when types are wrapped in Annotated.
    """

    class ComplexConfig(TypedDict):
        # Should be REQUIRED (no None)
        id: Annotated[int, "The ID"]

        # Should be OPTIONAL (has None), description should persist
        # Annotated wrapper shouldn't hide the UnionType from the nullable check
        filter: Annotated[str | None, "Optional filter"]

    def run_complex(cfg: ComplexConfig): pass

    tool = Tool.from_func(run_complex)
    props = tool.parameters["properties"]["cfg"]["properties"]
    reqs = tool.parameters["properties"]["cfg"]["required"]

    # 1. ID Check
    assert "id" in reqs
    assert props["id"]["type"] == "integer"
    assert props["id"]["description"] == "The ID"

    # 2. Filter Check
    # Crucial: Unwrapped Annotated to find UnionType, so it's NOT required
    assert "filter" not in reqs
    assert props["filter"]["type"] == "string"
    assert props["filter"]["description"] == "Optional filter"
