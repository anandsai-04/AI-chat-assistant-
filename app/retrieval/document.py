from typing import Any, Dict

from pydantic import BaseModel


class Document(BaseModel):
    """Represents a loaded piece of text (e.g., a file or a chunk)."""

    content: str
    metadata: Dict[str, Any] = {}
