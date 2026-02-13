import logging

from dinkleberg.fastapi import di
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from big_talk import BigTalk, Message

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/talk', tags=['talk'])


@router.get('/')
async def stream_talk(request: Request, bt=di(BigTalk)):
    # last_event_id = request.headers.get('Last-Event-ID')

    async def event_generator():
        async for message in bt.stream(model='anthropic/claude-haiku-4-5',
                                       messages=[Message(role='user', content='Write a haiku about the sea.')]):
            if await request.is_disconnected():
                logger.info('Client disconnected, stopping stream')
                break

            yield f'data: {message["content"]}\n\n'

    return StreamingResponse(event_generator(), media_type='text/event-stream')
