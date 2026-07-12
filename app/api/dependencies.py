from functools import lru_cache

from fastapi import Depends

from app.core.config import Settings, settings
from app.llm.factory import get_llm_client
from app.llm.interfaces import LLMInterface
from app.retrieval.vector_store import VectorStore
from app.services.chat import ChatService


def get_settings() -> Settings:
    """Dependency to inject the settings object."""
    return settings


def get_llm(settings: Settings = Depends(get_settings)) -> LLMInterface:
    """Dependency to inject the LLM client."""
    return get_llm_client(settings)


@lru_cache()
def get_vector_store() -> VectorStore:
    """Dependency to inject the VectorStore as a singleton."""
    return VectorStore()


def get_chat_service(
    llm: LLMInterface = Depends(get_llm),
    vector_store: VectorStore = Depends(get_vector_store),
) -> ChatService:
    """Dependency to inject the Chat Service."""
    return ChatService(llm_client=llm, vector_store=vector_store)
