from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from datetime import datetime
from passlib.context import CryptContext
import logging

from app.models.conversations import (
    Conversation, Message, FunctionCall,
    MessageDirection, Intent, FunctionCallStatus
)
from app.models.users import User
from app.schemas.chat import MessageCreate, ConversationCreate
from app.schemas.users import UserCreate
from app.core.exceptions import ResourceNotFoundError, UserAlreadyExistsError
from app.services.nlu import NLUService
from app.services.users import UserService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.nlu = NLUService()
        self.user_service = UserService(db)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def detect_intent_and_params(self, message: str) -> tuple[Intent, float, Dict[str, Any]]:
        # Use NLU service to detect intent
        intent, confidence = self.nlu.detect_intent(message)
        
        # Extract parameters based on detected intent
        params = self.nlu.extract_parameters(message, intent)
        
        return intent, confidence, params

    def get_function_for_intent(self, intent: Intent) -> Optional[Dict[str, Any]]:
        intent_to_function = {
            Intent.PRODUCT_SEARCH: {
                "name": "search_products",
                "description": "Search for products in the catalog",
                "parameters": {
                    "query": "string",
                    "category": "string?",
                    "max_price": "float?"
                }
            },
            Intent.ORDER_STATUS: {
                "name": "get_order_status",
                "description": "Get the status of an order",
                "parameters": {
                    "order_id": "integer"
                }
            },
            Intent.REGISTER_USER: {
                "name": "create_user",
                "description": "Create a new user account",
                "parameters": {
                    "name": "string",
                    "email": "string",
                    "password": "string"
                }
            }
        }
        return intent_to_function.get(intent)

    def create_conversation(self, conversation_data: ConversationCreate) -> Conversation:
        conversation = Conversation(
            user_id=conversation_data.user_id,
            active=True
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def get_conversation(self, conversation_id: int) -> Conversation:
        conversation = self.db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        if not conversation:
            raise ResourceNotFoundError("Conversation", conversation_id)
        return conversation

    def send_message(self, conversation_id: int, message_data: MessageCreate) -> Message:
        conversation = self.get_conversation(conversation_id)
        
        # Detect intent and extract parameters
        intent, confidence, params = self.detect_intent_and_params(message_data.content)
        
        # Log the detected intent and parameters
        logger.info(f"Detected intent: {intent} with confidence: {confidence}")
        logger.info(f"Extracted parameters: {params}")
        
        # Create user message
        user_message = Message(
            conversation_id=conversation_id,
            content=message_data.content,
            direction=MessageDirection.INCOMING,
            intent=intent,
            confidence_score=confidence
        )
        self.db.add(user_message)
        
        # Generate bot response
        response_content = self.generate_response(intent, message_data.content, params)
        bot_message = Message(
            conversation_id=conversation_id,
            content=response_content,
            direction=MessageDirection.OUTGOING
        )
        self.db.add(bot_message)
        
        # Check if we need to make a function call
        function_info = self.get_function_for_intent(intent)
        if function_info and all(params.get(param) for param in function_info["parameters"].keys()):
            try:
                # Execute the function based on intent
                if intent == Intent.REGISTER_USER:
                    logger.info(f"Creating user with parameters: {params}")
                    user_data = UserCreate(
                        name=params['name'],
                        email=params['email'],
                        password=params['password']
                    )
                    user = self.user_service.create_user(user_data)
                    logger.info(f"User created successfully with ID: {user.id}")
                    function_result = {"user_id": user.id, "email": user.email}
                    status = FunctionCallStatus.COMPLETED
                else:
                    # Handle other function calls here
                    function_result = None
                    status = FunctionCallStatus.PENDING

                function_call = FunctionCall(
                    message_id=user_message.id,
                    function_name=function_info["name"],
                    parameters=params,
                    result=function_result,
                    status=status,
                    completed_at=datetime.utcnow() if status == FunctionCallStatus.COMPLETED else None
                )
                self.db.add(function_call)

                # Update bot message if function was successful
                if status == FunctionCallStatus.COMPLETED:
                    if intent == Intent.REGISTER_USER:
                        bot_message.content = f"Great! Your account has been created successfully. Welcome, {user.name}!"
            except Exception as e:
                logger.error(f"Error during function call: {str(e)}")
                function_call = FunctionCall(
                    message_id=user_message.id,
                    function_name=function_info["name"],
                    parameters=params,
                    status=FunctionCallStatus.FAILED,
                    error_message=str(e),
                    completed_at=datetime.utcnow()
                )
                self.db.add(function_call)
                bot_message.content = f"I'm sorry, there was an error: {str(e)}"
        
        self.db.commit()
        self.db.refresh(user_message)
        self.db.refresh(bot_message)
        return user_message, bot_message

    def generate_response(self, intent: Intent, message: str, params: Dict[str, Any]) -> str:
        if intent == Intent.GREETING:
            return "Hello! How can I assist you today?"
        
        elif intent == Intent.PRODUCT_SEARCH:
            category = params.get('category', 'products')
            max_price = params.get('max_price')
            if max_price:
                return f"I'll help you find {category} under ${max_price}. Let me search our catalog..."
            return f"I'll help you find {category}. Let me search our catalog..."
        
        elif intent == Intent.ORDER_STATUS:
            order_id = params.get('order_id')
            if order_id:
                return f"Let me check the status of order #{order_id}..."
            return "Could you please provide your order number?"
        
        elif intent == Intent.HELP:
            return "I'm here to help! You can ask me about products, check order status, or get general assistance."
        
        elif intent == Intent.REGISTER_USER:
            missing_params = []
            if not params.get('name'):
                missing_params.append("name")
            if not params.get('email'):
                missing_params.append("email")
            if not params.get('password'):
                missing_params.append("password")
            
            if missing_params:
                return f"To complete your registration, I need your {', '.join(missing_params)}. Could you please provide them?"
            
            return "I'll help you create your account. Let me process your registration..."
        
        return "I'm not sure I understand. Could you please rephrase that?"

    def end_conversation(self, conversation_id: int) -> Conversation:
        conversation = self.get_conversation(conversation_id)
        conversation.active = False
        conversation.ended_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(conversation)
        return conversation 