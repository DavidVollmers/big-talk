from typing import TypedDict, Literal, Any, Union, TypeAlias, Sequence, Optional


class ToolUse(TypedDict):
    type: Literal['tool_use']
    id: str
    name: str
    params: dict[str, Any]
    metadata: Optional[dict[str, Any]]


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
    content: str


class ToolMessage(TypedDict):
    id: str
    parent_id: str
    role: Literal['tool']
    content: Sequence[ToolResult]


class AppMessage(TypedDict):
    id: str
    parent_id: Optional[str]
    role: Literal['app']
    type: str
    content: Any


AssistantContentBlock: TypeAlias = Union[Text, Thinking, ToolUse]


class AssistantMessage(TypedDict):
    id: str
    parent_id: str
    role: Literal['assistant']
    content: Sequence[AssistantContentBlock]
    is_aggregate: bool


InputMessage: TypeAlias = Union[UserMessage, SystemMessage, ToolMessage]

OutputMessage: TypeAlias = Union[AssistantMessage, AppMessage]

Message: TypeAlias = Union[InputMessage, OutputMessage]
