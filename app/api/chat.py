from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from pydantic import BaseModel

from app.core.chatbot import Chatbot, ChatResponse

router = APIRouter()
chatbot = Chatbot()

class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any] = {}

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and return a response
    """
    try:
        response = await chatbot.process_message(request.message)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history():
    """
    Get the conversation history
    """
    return {"history": chatbot.conversation_history}

@router.delete("/history")
async def clear_history():
    """
    Clear the conversation history
    """
    chatbot.conversation_history.clear()
    return {"message": "Conversation history cleared"} 