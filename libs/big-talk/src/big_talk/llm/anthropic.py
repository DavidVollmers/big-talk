from typing import TYPE_CHECKING, Iterable, AsyncGenerator

from anthropic import Omit, omit
from anthropic.types import MessageParam

from .llm import LLM
from ..message import Message

if TYPE_CHECKING:
    import anthropic


class Anthropic(LLM):
    def __init__(self):
        try:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic()
        except ImportError:
            raise ImportError(
                'The "anthropic" package is required to use the AnthropicProvider. '
                'Please install it with "pip install big-talk-ai[anthropic]".'
            )

    @staticmethod
    def _transform_messages(messages: Iterable[Message]) -> tuple[str | Omit, list[MessageParam]]:
        system_prompts = []
        transformed_messages = []
        for message in messages:
            if message['role'] == 'system':
                system_prompts.append(message['content'])
            else:
                # noinspection PyTypeChecker
                transformed_messages.append(MessageParam(
                    role=message['role'],
                    content=message['content']
                ))

        system_prompt = '\n'.join(system_prompts) if system_prompts else omit
        return system_prompt, transformed_messages

    async def count_tokens(self, model: str, messages: Iterable[Message], **kwargs) -> int:
        system, transformed_messages = self._transform_messages(messages)
        result = await self._client.messages.count_tokens(model=model, system=system, messages=transformed_messages,
                                                          **kwargs)
        return result.input_tokens

    async def stream(self, model: str, messages: Iterable[Message], **kwargs) -> AsyncGenerator[Message, None]:
        system, transformed_messages = self._transform_messages(messages)
        async with self._client.messages.stream(model=model, system=system, messages=transformed_messages,
                                                **kwargs) as stream:
            async for chunk in stream:
                if chunk.type == 'content_block_stop' and chunk.content_block.type == 'text':
                    yield Message(role='assistant', content=chunk.content_block.text)
