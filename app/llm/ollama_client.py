import logging
from typing import Any, Dict, List

from ollama import AsyncClient

from app.core.exceptions import AIException
from app.llm.interfaces import LLMInterface

logger = logging.getLogger("ai_assistant")


class OllamaLLM(LLMInterface):
    """Ollama implementation of the LLM interface."""

    def __init__(self, base_url: str, model: str):
        self.client = AsyncClient(host=base_url)
        self.model = model

    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        try:
            model_to_use = kwargs.pop("model", self.model)
            response = await self.client.chat(
                model=model_to_use, messages=messages, **kwargs
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise AIException(
                f"Failed to communicate with local LLM: {str(e)}", status_code=502
            )
