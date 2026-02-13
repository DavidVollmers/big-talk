import pytest


@pytest.mark.asyncio
async def test_chat(client):
    response = await client.get('/chat/')

    assert response.status_code == 200
    assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'

    events = []
    for line in response.iter_lines():
        if line:
            events.append(line)

    assert ''.join(
        events) == ('id: 1'
                    'data: Hello, this is a streamed response!'
                    ''
                    'id: 2'
                    'data: This is another message in the stream.'
                    ''
                    '')
