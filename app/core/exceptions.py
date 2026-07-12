import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("ai_assistant")


class AIException(Exception):
    """Base exception for AI Assistant errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code


def setup_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AIException)
    async def ai_exception_handler(request: Request, exc: AIException):
        logger.error(f"AI Exception on {request.url}: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )
