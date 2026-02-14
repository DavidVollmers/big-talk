from typing import TypedDict, Literal, Any, Union, TypeAlias, Sequence


class ToolUse(TypedDict):
    type: Literal['tool_use']
    id: str
    name: str
    params: dict[str, Any]


class ToolResult(TypedDict):
    type: Literal['tool_result']
    tool_use_id: str
    result: str
    is_error: bool


class Text(TypedDict):
    type: Literal['text']
    text: str


class Thinking(TypedDict):
    type: Literal['thinking']
    thinking: str
    signature: str


class UserMessage(TypedDict):
    id: str
    role: Literal['user']
    content: str


class SystemMessage(TypedDict):
    role: Literal['system']
    content: Union[str, Sequence[ToolResult]]


AssistantContentBlock: TypeAlias = Union[Text, Thinking, ToolUse]


class AssistantMessage(TypedDict):
    id: str
    parent_id: str
    role: Literal['assistant']
    content: Sequence[AssistantContentBlock]
    is_aggregate: bool


Message: TypeAlias = Union[UserMessage, SystemMessage, AssistantMessage]
