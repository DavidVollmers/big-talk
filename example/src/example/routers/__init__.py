from fastapi import APIRouter

from .talk_router import router as talk_router

api_router = APIRouter()

api_router.include_router(talk_router)
