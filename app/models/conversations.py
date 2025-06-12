from datetime import datetime
import enum
from sqlalchemy import Column, Integer, String, DateTime, Enum, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base

class MessageDirection(enum.Enum):
    INCOMING = "incoming"  # From user to bot
    OUTGOING = "outgoing"  # From bot to user

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    context = Column(JSON, nullable=True)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    content = Column(Text, nullable=False)
    direction = Column(Enum(MessageDirection), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    intent = Column(String(100), nullable=True)
    function_calls = Column(JSON, nullable=True)
    message_metadata = Column(JSON, nullable=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    function_calls_rel = relationship("FunctionCall", back_populates="message")

class FunctionCall(Base):
    __tablename__ = "function_calls"

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey("messages.id"))
    function_name = Column(String(255), nullable=False)
    parameters = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    message = relationship("Message", back_populates="function_calls_rel") 