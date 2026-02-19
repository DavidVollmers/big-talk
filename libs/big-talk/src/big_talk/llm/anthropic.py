from typing import Sequence, AsyncGenerator, Union
from uuid import uuid4

from anthropic import Omit, omit
from anthropic.types import MessageParam, ToolResultBlockParam, ThinkingBlockParam, TextBlockParam, ToolUseBlockParam, \
    ToolParam, ContentBlock

from .llm_provider import LLMProvider
from ..tool import Tool
from ..message import Message, AssistantContentBlock, ToolUse, Text, AssistantMessage, Thinking


class AnthropicProvider(LLMProvider):
    def __init__(self, **kwargs):
        try:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(**kwargs)
        except ImportError:
            raise ImportError(
                'The "anthropic" package is required to use the AnthropicProvider. '
                'Please install it with "pip install big-talk-ai[anthropic]".'
            )

    async def close(self):
        await self._client.close()

    async def count_tokens(self, model: str, messages: Sequence[Message], tools: Sequence[Tool], **kwargs) -> int:
        system, converted, _ = self._convert_messages(messages)
        tool_params = self._convert_tools(tools)
        result = await self._client.messages.count_tokens(model=model, system=system, messages=converted,
                                                          tools=tool_params, **kwargs)
        return result.input_tokens

    async def send(self, model: str, messages: Sequence[Message], tools: Sequence[Tool], **kwargs) -> AssistantMessage:
        system, converted, last_user_message_id = self._convert_messages(messages)
        tool_params = self._convert_tools(tools)
        tool_map = {tool.name: tool for tool in tools}
        response = await self._client.messages.create(model=model, system=system, messages=converted, tools=tool_params,
                                                      **kwargs)
        content = [AnthropicProvider._to_block(block, tool_map) for block in response.content]
        return AssistantMessage(id=str(uuid4()), role='assistant', content=content, parent_id=last_user_message_id,
                                is_aggregate=True)

    async def stream(self, model: str, messages: Sequence[Message], tools: Sequence[Tool], **kwargs) \
            -> AsyncGenerator[AssistantMessage, None]:
        system, converted, last_user_message_id = self._convert_messages(messages)
        tool_params = self._convert_tools(tools)
        tool_map = {tool.name: tool for tool in tools}
        async with self._client.messages.stream(model=model,
                                                system=system,
                                                messages=converted,
                                                tools=tool_params,
                                                **kwargs) as stream:
            message_id = str(uuid4())
            blocks: list[AssistantContentBlock] = []
            async for chunk in stream:
                if chunk.type == 'message_stop':
                    yield AssistantMessage(id=message_id, role='assistant', content=blocks,
                                           parent_id=last_user_message_id, is_aggregate=True)
                    blocks = []
                    continue

                if chunk.type != 'content_block_stop':
                    continue

                block = self._to_block(chunk.content_block, tool_map)
                if not block:
                    continue

                blocks.append(block)
                yield AssistantMessage(id=message_id, role='assistant', content=[block], parent_id=last_user_message_id,
                                       is_aggregate=False)

    @staticmethod
    def _convert_tools(tools: Sequence[Tool]) -> list[ToolParam]:
        # noinspection PyTypeChecker
        return [
            ToolParam(
                name=tool.name,
                description=tool.description,
                input_schema=tool.parameters
            ) for tool in tools
        ]

    @staticmethod
    def _convert_messages(messages: Sequence[Message]) -> tuple[str | Omit, list[MessageParam], str]:
        system_parts = []
        converted = []
        last_user_message_id = None
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                system_parts.append(content)
            elif role == 'tool':
                converted.append(MessageParam(
                    role='user',
                    content=[ToolResultBlockParam(type='tool_result',
                                                  tool_use_id=block['tool_use_id'],
                                                  content=block['result'],
                                                  is_error=block['is_error']) for block in content]
                ))
            elif role == 'user':
                last_user_message_id = message['id']
                converted.append(MessageParam(
                    role='user',
                    content=content
                ))
            elif role == 'assistant':
                converted.append(MessageParam(
                    role='assistant',
                    content=[AnthropicProvider._from_block(block) for block in content]
                ))

        system = '\n'.join(system_parts) if system_parts else omit
        return system, converted, last_user_message_id

    @staticmethod
    def _from_block(block: AssistantContentBlock) -> Union[TextBlockParam, ThinkingBlockParam, ToolUseBlockParam]:
        match block:
            case {'type': 'text', 'text': text}:
                return TextBlockParam(type='text', text=text)
            case {'type': 'thinking', 'thinking': thinking, 'signature': signature}:
                return ThinkingBlockParam(type='thinking', thinking=thinking, signature=signature)
            case {'type': 'tool_use', 'id': tool_use_id, 'name': name, 'params': params}:
                return ToolUseBlockParam(type='tool_use', id=tool_use_id, name=name, input=params)
            case _:
                raise ValueError(f'Unsupported content block: {block}')

    @staticmethod
    def _to_block(block: ContentBlock, tool_map: dict[str, Tool]) -> AssistantContentBlock | None:
        if block.type == 'text':
            return Text(type='text', text=block.text)
        elif block.type == 'thinking':
            return Thinking(type='thinking', thinking=block.thinking, signature=block.signature)
        elif block.type == 'tool_use':
            metadata = tool_map[block.name].metadata if block.name in tool_map else None
            return ToolUse(type='tool_use', id=block.id, name=block.name, params=block.input, metadata=metadata)
        else:
            # TODO redacted thinking
            return None
