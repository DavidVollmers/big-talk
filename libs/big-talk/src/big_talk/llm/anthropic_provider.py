from typing import Sequence, AsyncGenerator, Union
from uuid import uuid4

from anthropic import Omit, omit
from anthropic.types import MessageParam, ToolResultBlockParam, ThinkingBlockParam, TextBlockParam, ToolUseBlockParam

from .llm_provider import LLMProvider
from ..message import Message, AssistantContentBlock, ToolResult, ToolUse, Text, AssistantMessage


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

    async def count_tokens(self, model: str, messages: Sequence[Message], **kwargs) -> int:
        system, converted, _ = self._convert_messages(messages)
        result = await self._client.messages.count_tokens(model=model, system=system, messages=converted, **kwargs)
        return result.input_tokens

    async def stream(self, model: str, messages: Sequence[Message], **kwargs) -> AsyncGenerator[Message, None]:
        system, converted, last_user_message_id = self._convert_messages(messages)
        async with self._client.messages.stream(model=model, system=system, messages=converted, **kwargs) as stream:
            message_id = str(uuid4())
            blocks: list[AssistantContentBlock] = []
            async for chunk in stream:
                if chunk.type == 'message_stop':
                    yield AssistantMessage(id=message_id, role='assistant', content=blocks,
                                           parent_id=last_user_message_id, is_aggregate=True)
                    blocks = []

                if chunk.type != 'content_block_stop':
                    continue

                if chunk.content_block.type == 'text':
                    block = Text(type='text', text=chunk.content_block.text)
                elif chunk.content_block.type == 'thinking':
                    block = ThinkingBlockParam(type='thinking',
                                               thinking=chunk.content_block.thinking,
                                               signature=chunk.content_block.signature)
                elif chunk.content_block.type == 'tool_use':
                    block = ToolUse(type='tool_use', id=chunk.content_block.id, name=chunk.content_block.name,
                                    params=chunk.content_block.input)
                else:
                    # TODO redacted thinking
                    continue

                blocks.append(block)
                yield AssistantMessage(id=message_id, role='assistant', content=[block], parent_id=last_user_message_id,
                                       is_aggregate=False)

    @staticmethod
    def _convert_messages(messages: Sequence[Message]) -> tuple[str | Omit, list[MessageParam], str]:
        system_parts = []
        converted = []
        last_user_message_id = None
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                if isinstance(content, str):
                    system_parts.append(content)
                else:
                    last_user_message_id = message['id']
                    converted.append(MessageParam(
                        role='user',
                        content=[AnthropicProvider._convert_tool_result(block) for block in content]
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
        return system, converted, last_user_message_id

    @staticmethod
    def _convert_tool_result(block: ToolResult) -> ToolResultBlockParam:
        match block:
            case {'type': 'tool_result', 'tool_use_id': tool_use_id, 'result': result, 'is_error': is_error}:
                return ToolResultBlockParam(type='tool_result', tool_use_id=tool_use_id, content=result,
                                            is_error=is_error)
            case _:
                raise ValueError(f'Expected tool result block, got: {block}')

    @staticmethod
    def _convert_block(block: AssistantContentBlock) -> Union[TextBlockParam, ThinkingBlockParam, ToolUseBlockParam]:
        match block:
            case {'type': 'text', 'text': text}:
                return TextBlockParam(type='text', text=text)
            case {'type': 'thinking', 'thinking': thinking, 'signature': signature}:
                return ThinkingBlockParam(type='thinking', thinking=thinking, signature=signature)
            case {'type': 'tool_use', 'id': tool_use_id, 'name': name, 'params': params}:
                return ToolUseBlockParam(type='tool_use', id=tool_use_id, name=name, input=params)
            case _:
                raise ValueError(f'Unsupported content block: {block}')
