from app.core.config import Settings
from app.llm.interfaces import LLMInterface
from app.llm.ollama_client import OllamaLLM


def get_llm_client(settings: Settings) -> LLMInterface:
    """Factory to return the configured LLM client.

    This abstracts away the underlying provider (Ollama, OpenAI, Anthropic).
    """
    return OllamaLLM(base_url=settings.OLLAMA_BASE_URL, model=settings.DEFAULT_MODEL)
