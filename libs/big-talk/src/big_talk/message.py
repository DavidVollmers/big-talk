from typing import TypedDict, Literal, Any, Union, TypeAlias, Iterable


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
    type: Literal['content']
    text: str


class Thinking(TypedDict):
    type: Literal['thinking']
    thinking: str


class UserMessage(TypedDict):
    role: Literal['user']
    content: str


class SystemMessage(TypedDict):
    role: Literal['system']
    content: Union[str, Iterable[ToolResult]]


class AssistantMessage(TypedDict):
    role: Literal['assistant']
    content: Iterable[Union[Text, Thinking, ToolUse]]


Message: TypeAlias = Union[UserMessage, SystemMessage, AssistantMessage]
