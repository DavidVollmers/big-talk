import pytest_asyncio
from asgi_lifespan import LifespanManager
from dinkleberg import DependencyConfigurator
from httpx import AsyncClient, ASGITransport

from example.main import api


@pytest_asyncio.fixture
async def app():
    async with LifespanManager(api) as manager:
        deps: DependencyConfigurator = api.state.dinkleberg

        # TODO add test dependencies here

        yield manager.app


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
