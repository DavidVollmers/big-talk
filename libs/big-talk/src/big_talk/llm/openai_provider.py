from typing import Sequence, AsyncGenerator

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam

from ..message import Message
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

    @staticmethod
    def _transform_messages(messages: Sequence[Message]) -> list[ChatCompletionMessageParam]:
        transformed_messages = []
        for message in messages:
            if message['role'] == 'user':
                transformed_messages.append(ChatCompletionUserMessageParam(
                    role='user',
                    content=message['content']
                ))
        return transformed_messages

    async def count_tokens(self, model: str, messages: Sequence[Message], **kwargs) -> int:
        # encoding = self._encoding_for_model(model)
        # TODO https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken/
        raise NotImplementedError(
            'Token counting is not implemented for OpenAIProvider yet. Please use the tiktoken library directly.')

    async def stream(self, model: str, messages: Sequence[Message], **kwargs) -> AsyncGenerator[Message, None]:
        transformed_messages = self._transform_messages(messages)
        response = await self._client.chat.completions.create(model=model, messages=transformed_messages, stream=True,
                                                              **kwargs)
        async with response as stream:
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield Message(role=chunk.choices[0].delta.role, content=chunk.choices[0].delta.content)
