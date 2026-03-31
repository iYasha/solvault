from fastapi import APIRouter

from sol.gateway.api.v1 import health, messages, ws

api_router = APIRouter(prefix="/v1")
api_router.include_router(health.router)
api_router.include_router(messages.router)
api_router.include_router(ws.router)
