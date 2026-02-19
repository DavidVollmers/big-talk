import json
from typing import Sequence, AsyncGenerator, TYPE_CHECKING
from uuid import uuid4

from openai.types.shared_params import FunctionDefinition
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionToolMessageParam, ChatCompletionUserMessageParam, ChatCompletionMessageFunctionToolCallParam, \
    ChatCompletionAssistantMessageParam, ChatCompletionFunctionToolParam
from openai.types.chat.chat_completion_message_function_tool_call_param import Function

from ..tool import Tool
from ..message import Message, AssistantContentBlock, Text, AssistantMessage, ToolUse
from .llm_provider import LLMProvider

if TYPE_CHECKING:
    from tiktoken import Encoding


class OpenAIProvider(LLMProvider):
    def __init__(self, **kwargs):
        try:
            from openai import AsyncOpenAI
            from tiktoken import encoding_for_model
            self._client = AsyncOpenAI(**kwargs)
            self._encoding_for_model = encoding_for_model
        except ImportError:
            raise ImportError(
                'The "openai" package is required to use the OpenAIProvider. '
                'Please install it with "pip install big-talk-ai[openai]".'
            )

    async def close(self):
        await self._client.close()

    async def count_tokens(self, model: str, messages: Sequence[Message], tools: Sequence[Tool], **kwargs) -> int:
        encoding = self._encoding_for_model(model)
        converted_messages, _ = self._convert_messages(messages)
        # TODO calculate tool tokens
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        return self._count_message_tokens(converted_messages, model, encoding)

    async def send(self, model: str, messages: Sequence[Message], tools: Sequence[Tool], **kwargs) -> AssistantMessage:
        converted, last_user_message_id = self._convert_messages(messages)
        tool_params, tool_map = self._convert_tools(tools)

        response = await self._client.chat.completions.create(model=model, messages=converted, tools=tool_params,
                                                              **kwargs)

        blocks = []
        for choice in response.choices:
            msg = choice.message
            if msg.content:
                blocks.append(Text(type='text', text=msg.content))
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    blocks.append(ToolUse(
                        type='tool_use',
                        id=tool_call.id,
                        name=tool_call.function.name,
                        params=json.loads(tool_call.function.arguments),
                        metadata=tool_map[
                            tool_call.function.name].metadata if tool_call.function.name in tool_map else None
                    ))

        return AssistantMessage(id=str(uuid4()), role='assistant', content=blocks, parent_id=last_user_message_id,
                                is_aggregate=True)

    async def stream(self, model: str, messages: Sequence[Message], tools: Sequence[Tool], **kwargs) \
            -> AsyncGenerator[AssistantMessage, None]:
        converted, last_user_message_id = self._convert_messages(messages)
        tool_params, tool_map = self._convert_tools(tools)

        text_buffer: list[str] = []
        current_tool_index: int | None = None
        current_tool_id: str = ''
        current_tool_name: str = ''
        current_tool_args: list[str] = []

        blocks: list[AssistantContentBlock] = []

        stream = await self._client.chat.completions.create(
            model=model,
            messages=converted,
            tools=tool_params,
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
                        prev_block = self._build_tool_use_block(current_tool_id, current_tool_name, current_tool_args,
                                                                tool_map)
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
            last_block = self._build_tool_use_block(current_tool_id, current_tool_name, current_tool_args, tool_map)
            blocks.append(last_block)
            yield AssistantMessage(
                id=message_id,
                role='assistant',
                content=[last_block],
                parent_id=last_user_message_id,
                is_aggregate=False
            )

        yield AssistantMessage(
            id=message_id,
            role='assistant',
            content=blocks,
            parent_id=last_user_message_id,
            is_aggregate=True
        )

    @staticmethod
    def _convert_tools(tools: Sequence[Tool]) -> tuple[list[ChatCompletionFunctionToolParam], dict[str, Tool]]:
        # noinspection PyTypeChecker
        tool_params = [
            ChatCompletionFunctionToolParam(
                type='function',
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters
                )
            ) for tool in tools
        ]
        tool_map = {tool.name: tool for tool in tools}
        return tool_params, tool_map

    @staticmethod
    def _build_tool_use_block(tool_id: str, tool_name: str, arg_parts: list[str], tool_map: dict[str, Tool]) -> ToolUse:
        metadata = tool_map[tool_name].metadata if tool_name in tool_map else None
        return ToolUse(
            type='tool_use',
            id=tool_id,
            name=tool_name,
            params=json.loads(''.join(arg_parts)),
            metadata=metadata
        )

    @staticmethod
    def _convert_messages(messages: Sequence[Message]) -> tuple[list[ChatCompletionMessageParam], str]:
        converted = []
        last_user_message_id = None

        for message in messages:
            role = message['role']
            content = message['content']

            if role == 'system':
                converted.append(ChatCompletionSystemMessageParam(
                    role='system',
                    content=content
                ))

            elif role == 'tool':
                last_user_message_id = message['id']
                for block in content:
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

    @staticmethod
    def _count_message_tokens(messages: list[ChatCompletionMessageParam], model: str, encoding: 'Encoding') -> int:
        # Constants for message overhead
        if model.startswith('gpt-4o'):
            tokens_per_message = 3
            tokens_per_name = 1
        elif model.startswith('gpt-3.5-turbo') or model.startswith('gpt-4'):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            # Conservative default
            tokens_per_message = 3
            tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == 'content' and value:
                    num_tokens += len(encoding.encode(value))
                elif key == 'tool_calls':
                    for tool in value:
                        # Overhead for tool calls within a message
                        num_tokens += len(encoding.encode(tool['function']['name']))
                        num_tokens += len(encoding.encode(tool['function']['arguments']))
                        # There is usually a small overhead for the tool_calls structure itself,
                        # but it's not officially documented. +3 is a safe buffer.
                        num_tokens += 3
                elif key == 'name':
                    num_tokens += len(encoding.encode(value))
                    num_tokens += tokens_per_name
                elif key == 'role':
                    pass  # Role is handled by tokens_per_message overhead

        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        return num_tokens
