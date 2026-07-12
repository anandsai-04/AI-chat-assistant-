from typing import Dict, List


class PromptBuilder:
    """Utility to build prompt sequences for LLMs."""
    
    @staticmethod
    def build_chat_prompt(
        system_prompt: str, user_message: str
    ) -> List[Dict[str, str]]:
        """Constructs the standard system/user message list."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    
    @staticmethod
    def get_default_system_prompt() -> str:
        """Returns the base system prompt for the AI Assistant."""
        return (
            "You are a Senior AI Engineering Assistant. "
            "Provide helpful, accurate, and concise answers to software engineering "
            "and AI-related questions. Always structure your responses clearly."
        )
