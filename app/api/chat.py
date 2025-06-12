from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.models.database import get_db
from app.schemas.chat import (
    MessageCreate,
    MessageResponse,
    ConversationCreate,
    ConversationResponse,
    ChatTurnResponse
)
from app.services.chat import ChatService

router = APIRouter()

@router.post("/conversations/", response_model=ConversationResponse)
def create_conversation(
    conversation: ConversationCreate,
    db: Session = Depends(get_db)
):
    """
    Start a new conversation.
    """
    chat_service = ChatService(db)
    return chat_service.create_conversation(conversation)

@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a conversation by ID with all its messages.
    """
    chat_service = ChatService(db)
    return chat_service.get_conversation(conversation_id)

@router.post("/conversations/{conversation_id}/messages/", response_model=ChatTurnResponse)
def send_message(
    conversation_id: int,
    message: MessageCreate,
    db: Session = Depends(get_db)
):
    """
    Send a message in a conversation.
    The service will:
    1. Detect the message intent
    2. Generate an appropriate response
    3. Make function calls if needed
    """
    chat_service = ChatService(db)
    user_msg, bot_msg = chat_service.send_message(conversation_id, message)
    return ChatTurnResponse(user_message=user_msg, bot_message=bot_msg)

@router.post("/conversations/{conversation_id}/end", response_model=ConversationResponse)
def end_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """
    End a conversation.
    """
    chat_service = ChatService(db)
    return chat_service.end_conversation(conversation_id) 