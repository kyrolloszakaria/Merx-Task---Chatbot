from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.conversations import MessageDirection, Intent, FunctionCallStatus

class MessageBase(BaseModel):
    content: str = Field(..., min_length=1, max_length=4096)

class MessageCreate(MessageBase):
    pass

class FunctionCallResponse(BaseModel):
    function_name: str
    parameters: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    status: FunctionCallStatus
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        orm_mode = True

class MessageResponse(MessageBase):
    id: int
    direction: MessageDirection
    timestamp: datetime
    intent: Optional[Intent]
    confidence_score: Optional[float]
    function_calls: Optional[List[FunctionCallResponse]]

    class Config:
        orm_mode = True

class ConversationCreate(BaseModel):
    user_id: int

class ConversationResponse(BaseModel):
    id: int
    user_id: int
    started_at: datetime
    ended_at: Optional[datetime]
    active: bool
    messages: List[MessageResponse]

    class Config:
        orm_mode = True

class ChatTurnResponse(BaseModel):
    user_message: MessageResponse
    bot_message: MessageResponse 