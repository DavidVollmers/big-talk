from typing import Iterable, AsyncGenerator, Union

from anthropic import Omit, omit
from anthropic.types import MessageParam, ToolResultBlockParam, ThinkingBlockParam

from .llm_provider import LLMProvider
from ..message import Message, Text, Thinking, ToolUse


class AnthropicProvider(LLMProvider):
    def __init__(self):
        try:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic()
        except ImportError:
            raise ImportError(
                'The "anthropic" package is required to use the AnthropicProvider. '
                'Please install it with "pip install big-talk-ai[anthropic]".'
            )

    async def close(self):
        await self._client.close()

    async def count_tokens(self, model: str, messages: Iterable[Message], **kwargs) -> int:
        system, converted = self._convert_messages(messages)
        result = await self._client.messages.count_tokens(model=model, system=system, messages=converted, **kwargs)
        return result.input_tokens

    async def stream(self, model: str, messages: Iterable[Message], **kwargs) -> AsyncGenerator[Message, None]:
        system, converted = self._convert_messages(messages)
        async with self._client.messages.stream(model=model, system=system, messages=converted, **kwargs) as stream:
            async for chunk in stream:
                if chunk.type == 'content_block_stop' and chunk.content_block.type == 'text':
                    yield Message(role='assistant', content=chunk.content_block.text)

    @staticmethod
    def _convert_messages(messages: Iterable[Message]) -> tuple[str | Omit, list[MessageParam]]:
        system_parts = []
        converted = []
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                if isinstance(content, str):
                    system_parts.append(content)
                else:
                    converted.append(MessageParam(
                        role='user',
                        content=[
                            ToolResultBlockParam(
                                type='tool_result',
                                tool_use_id=block['tool_use_id'],
                                content=block['result'],
                                is_error=block['is_error']
                            ) for block in content if block['type'] == 'tool_result'
                        ]
                    ))
            elif role == 'user':
                converted.append(MessageParam(
                    role='user',
                    content=content
                ))
            elif role == 'assistant':
                converted.append(MessageParam(
                    role='assistant',
                    content=[AnthropicProvider._convert_block(block) for block in content]
                ))

        system = '\n'.join(system_parts) if system_parts else omit
        return system, converted

    @staticmethod
    def _convert_block(block: Union[Text, Thinking, ToolUse]) -> Union[ThinkingBlockParam]:
        pass
