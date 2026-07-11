from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings
from app.core.config import Settings

router = APIRouter()

@router.get("/health")
def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint to ensure API is running."""
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "version": "0.1.0"
    }
