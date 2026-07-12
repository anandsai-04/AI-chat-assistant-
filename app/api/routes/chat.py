from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import get_chat_service
from app.services.chat import ChatService

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Endpoint for chatting with the AI Assistant."""
    reply = await chat_service.process_message(request.message)
    return ChatResponse(response=reply)
