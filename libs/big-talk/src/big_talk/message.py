from typing import TypedDict, Literal


class Message(TypedDict):
    role: Literal['user', 'system', 'assistant']
    content: str
