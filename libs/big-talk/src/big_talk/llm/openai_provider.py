import json
from typing import Sequence, AsyncGenerator
from uuid import uuid4

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionToolMessageParam, ChatCompletionUserMessageParam, ChatCompletionMessageFunctionToolCallParam, \
    ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_message_function_tool_call_param import Function

from .. import ToolUse
from ..message import Message, AssistantContentBlock, Text, AssistantMessage
from .llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self):
        try:
            from openai import AsyncOpenAI
            from tiktoken import encoding_for_model
            self._client = AsyncOpenAI()
            self._encoding_for_model = encoding_for_model
        except ImportError:
            raise ImportError(
                'The "openai" package is required to use the OpenAIProvider. '
                'Please install it with "pip install big-talk-ai[openai]".'
            )

    async def close(self):
        await self._client.close()

    async def count_tokens(self, model: str, messages: Sequence[Message], **kwargs) -> int:
        # encoding = self._encoding_for_model(model)
        # TODO https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken/
        raise NotImplementedError(
            'Token counting is not implemented for OpenAIProvider yet. Please use the tiktoken library directly.')

    async def stream(self, model: str, messages: Sequence[Message], **kwargs) -> AsyncGenerator[Message, None]:
        converted, last_user_message_id = self._convert_messages(messages)

        text_buffer: list[str] = []
        current_tool_index: int | None = None
        current_tool_id: str = ''
        current_tool_name: str = ''
        current_tool_args: list[str] = []

        blocks: list[AssistantContentBlock] = []

        stream = await self._client.chat.completions.create(
            model=model,
            messages=converted,
            stream=True,
            **kwargs
        )

        message_id = str(uuid4())
        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                text_buffer.append(delta.content)

            if delta.tool_calls:
                if text_buffer:
                    full_text = ''.join(text_buffer)
                    block = Text(type='text', text=full_text)
                    blocks.append(block)

                    yield AssistantMessage(
                        id=message_id,
                        role='assistant',
                        content=[block],
                        parent_id=last_user_message_id,
                        is_aggregate=False
                    )
                    text_buffer = []

                for tool_chunk in delta.tool_calls:
                    idx = tool_chunk.index

                    if current_tool_index is not None and idx != current_tool_index:
                        prev_block = self._build_tool_use_block(current_tool_id, current_tool_name, current_tool_args)
                        blocks.append(prev_block)

                        yield AssistantMessage(
                            id=message_id,
                            role='assistant',
                            content=[prev_block],
                            parent_id=last_user_message_id,
                            is_aggregate=False
                        )

                        current_tool_id = ''
                        current_tool_name = ''
                        current_tool_args = []

                    current_tool_index = idx
                    if tool_chunk.id:
                        current_tool_id = tool_chunk.id
                    if tool_chunk.function.name:
                        current_tool_name = tool_chunk.function.name
                    if tool_chunk.function.arguments:
                        current_tool_args.append(tool_chunk.function.arguments)

        if text_buffer:
            full_text = ''.join(text_buffer)
            block = Text(type='text', text=full_text)
            blocks.append(block)
            yield AssistantMessage(
                id=message_id,
                role='assistant',
                content=[block],
                parent_id=last_user_message_id,
                is_aggregate=False
            )

        if current_tool_index is not None:
            last_block = self._build_tool_use_block(current_tool_id, current_tool_name, current_tool_args)
            blocks.append(last_block)
            yield AssistantMessage(
                id=message_id,
                role='assistant',
                content=[last_block],
                parent_id=last_user_message_id,
                is_aggregate=False
            )

        # 3. Yield Aggregate
        yield AssistantMessage(
            id=message_id,
            role='assistant',
            content=blocks,
            parent_id=last_user_message_id,
            is_aggregate=True
        )

    @staticmethod
    def _build_tool_use_block(tool_id: str, tool_name: str, arg_parts: list[str]) -> ToolUse:
        return ToolUse(
            type='tool_use',
            id=tool_id,
            name=tool_name,
            params=json.loads(''.join(arg_parts))
        )

    @staticmethod
    def _convert_messages(messages: Sequence[Message]) -> tuple[list[ChatCompletionMessageParam], str]:
        converted = []
        last_user_message_id = None

        for message in messages:
            role = message['role']
            content = message['content']

            if role == 'system':
                if isinstance(content, str):
                    converted.append(ChatCompletionSystemMessageParam(
                        role='system',
                        content=content
                    ))
                else:
                    last_user_message_id = message['id']
                    for block in content:
                        if block['type'] != 'tool_result':
                            raise ValueError(f'Expected tool result block, got: {block}')
                        converted.append(ChatCompletionToolMessageParam(
                            role='tool',
                            tool_call_id=block['tool_use_id'],
                            content=block['result'],
                        ))

            elif role == 'user':
                converted.append(ChatCompletionUserMessageParam(
                    role='user',
                    content=content
                ))

            elif role == 'assistant':
                text_parts: list[str] = []
                tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = []

                for block in content:
                    if block['type'] == 'text':
                        text_parts.append(block['text'])
                    elif block['type'] == 'tool_use':
                        tool_calls.append(ChatCompletionMessageFunctionToolCallParam(
                            id=block['id'],
                            type='function',
                            function=Function(
                                name=block['name'],
                                arguments=json.dumps(block['params'])
                            )
                        ))

                converted.append(ChatCompletionAssistantMessageParam(
                    role='assistant',
                    content='\n'.join(text_parts) if text_parts else None,
                    tool_calls=tool_calls if tool_calls else None
                ))

        return converted, last_user_message_id
