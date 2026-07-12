from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMInterface(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Send a list of messages and receive a string response."""
        pass
