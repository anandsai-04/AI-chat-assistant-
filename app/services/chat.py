from app.llm.interfaces import LLMInterface
from app.prompts.builder import PromptBuilder


class ChatService:
    """Business logic for handling chat interactions."""
    
    def __init__(self, llm_client: LLMInterface):
        self.llm = llm_client
        
    async def process_message(self, user_message: str) -> str:
        """Process a user message and get an AI response."""
        system_prompt = PromptBuilder.get_default_system_prompt()
        messages = PromptBuilder.build_chat_prompt(system_prompt, user_message)
        
        response = await self.llm.chat(messages=messages)
        return response
