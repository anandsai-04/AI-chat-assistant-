from fastapi import FastAPI

from app.api.routes import health
from app.core.config import settings
from app.core.exceptions import setup_exception_handlers
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    """Application factory for FastAPI."""
    
    # 1. Setup Logging
    setup_logging()
    
    # 2. Initialize App
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="0.1.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )
    
    # 3. Setup Exception Handlers
    setup_exception_handlers(app)
    
    # 4. Include Routers
    app.include_router(health.router, prefix=settings.API_V1_STR, tags=["health"])
    
    return app

app = create_app()
