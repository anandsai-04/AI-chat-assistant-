from typing import Dict, List


class PromptBuilder:
    """Utility to build prompt sequences for LLMs."""

    @staticmethod
    def build_chat_prompt(
        system_prompt: str, user_message: str, context: str = None
    ) -> List[Dict[str, str]]:
        """Constructs the standard system/user message list with optional context."""

        final_user_message = user_message
        if context:
            final_user_message = (
                f"Context Information:\n"
                f"---------------------\n"
                f"{context}\n"
                f"---------------------\n\n"
                f"Question: {user_message}\n\n"
                f"Answer the question based ONLY on the context provided above. "
                f"If the answer is not in the context, say 'I don't know based on the provided context'."
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_user_message},
        ]

    @staticmethod
    def get_default_system_prompt() -> str:
        """Returns the base system prompt for the AI Assistant."""
        return (
            "You are an AI Domain Expert. "
            "Provide helpful, accurate, and concise answers based on the retrieved context. "
            "Always structure your responses clearly."
        )
