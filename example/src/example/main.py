from contextlib import asynccontextmanager

from fastapi import FastAPI
from dinkleberg import DependencyConfigurator

from .routers import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    di = app.state.dinkleberg = DependencyConfigurator()



    yield

    await di.close()


api = FastAPI(lifespan=lifespan)

api.include_router(api_router)
