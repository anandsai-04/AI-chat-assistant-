from app.llm.interfaces import LLMInterface
from app.prompts.builder import PromptBuilder
from app.retrieval.vector_store import VectorStore


class ChatService:
    """Business logic for handling chat interactions."""

    def __init__(self, llm_client: LLMInterface, vector_store: VectorStore = None):
        self.llm = llm_client
        self.vector_store = vector_store

    async def process_message(self, user_message: str) -> str:
        """Process a user message using RAG and get an AI response."""
        context_str = None

        if self.vector_store:
            try:
                # Retrieve top 3 relevant chunks
                docs = self.vector_store.search(user_message, n_results=3)
                if docs:
                    context_str = "\n\n".join([doc.content for doc in docs])
            except Exception as e:
                # If search fails (e.g., empty DB), just proceed without context
                print(f"Warning: Vector search failed: {e}")

        system_prompt = PromptBuilder.get_default_system_prompt()
        messages = PromptBuilder.build_chat_prompt(
            system_prompt, user_message, context_str
        )

        response = await self.llm.chat(messages=messages)
        return response
