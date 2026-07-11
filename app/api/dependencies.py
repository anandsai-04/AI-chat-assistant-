from app.core.config import Settings, settings


def get_settings() -> Settings:
    """Dependency to inject the settings object."""
    return settings
