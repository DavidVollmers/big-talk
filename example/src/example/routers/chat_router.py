import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/chat', tags=['chat'])


@router.get('/')
async def stream_chat(request: Request):
    last_event_id = request.headers.get('Last-Event-ID')

    async def event_generator():
        yield 'id: 1\n'
        yield 'data: Hello, this is a streamed response!\n\n'
        yield 'id: 2\n'
        yield 'data: This is another message in the stream.\n\n'

    return StreamingResponse(event_generator(), media_type='text/event-stream')
