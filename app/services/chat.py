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
from app.schemas.users import UserCreate, UserModify
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

    def update_conversation_context(self, conversation: Conversation, intent: Intent, params: Dict[str, Any] = None) -> None:
        """Update the conversation context with the current intent and parameters"""
        if intent != Intent.UNKNOWN:
            context = {
                'current_intent': intent.value,
                'params': params or {},
                'updated_at': datetime.utcnow().isoformat()
            }
            conversation.context = context
            self.db.add(conversation)
            self.db.commit()
    
    def clear_conversation_context(self, conversation: Conversation) -> None:
        """Clear the conversation context"""
        conversation.context = None
        self.db.add(conversation)
        self.db.commit()

    def get_current_context(self, conversation: Conversation) -> Optional[Dict[str, Any]]:
        """Get the current conversation context"""
        if not conversation.context:
            return None
        
        # Check if context is too old (e.g., more than 5 minutes)
        if conversation.context.get('updated_at'):
            last_update = datetime.fromisoformat(conversation.context['updated_at'])
            if (datetime.utcnow() - last_update).total_seconds() > 300:  # 5 minutes
                self.clear_conversation_context(conversation)
                return None
        
        return conversation.context

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
            Intent.MODIFY_USER: {
                "name": "modify_user",
                "description": "Update user account information",
                "parameters": {
                    "user_id": "integer",
                    "user_data": {
                        "name": "string?",
                        "email": "string?",
                        "password": "string?"
                    }
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
        self.conversation = conversation
        
        # Get current context
        current_context = self.get_current_context(conversation)
        
        # Detect intent and extract parameters
        intent, confidence = self.nlu.detect_intent(message_data.content)
        logger.info(f"Detected intent: {intent} with confidence: {confidence}")
        params = self.nlu.extract_parameters(message_data.content, intent)

        # Handle context and low confidence cases
        if confidence < 0.3 and current_context:
            stored_intent = Intent(current_context.get('current_intent'))
            # Check if we're in the middle of a multi-step interaction
            if stored_intent in [Intent.MODIFY_USER, Intent.PRODUCT_SEARCH, Intent.ORDER_STATUS]:
                intent = stored_intent
                # Merge current parameters with context parameters
                context_params = current_context.get('params', {})
                params = self.nlu.extract_parameters(message_data.content, intent)
                logger.info(f"Extracted parameters: {params}")
                params = {**context_params, **params}
                logger.info(f"Merged parameters: {params}")
                # Keep the existing context since we're continuing the conversation
                conversation.context = current_context
                self.db.add(conversation)
            else:
                # If stored intent isn't for multi-step interaction, treat as unknown
                intent = Intent.UNKNOWN
        
        # Update conversation context for new intents
        if intent != Intent.UNKNOWN:
            self.update_conversation_context(conversation, intent, params)
        
        # Create user message
        user_message = Message(
            conversation_id=conversation_id,
            content=message_data.content,
            direction=MessageDirection.INCOMING,
            intent=intent,
            confidence_score=confidence
        )
        self.db.add(user_message)
        
        # Create bot message
        bot_message = Message(
            conversation_id=conversation_id,
            content="",  # We'll set this later
            direction=MessageDirection.OUTGOING
        )
        
        # Check if we need to make a function call
        function_info = self.get_function_for_intent(intent)
        if function_info:
            try:
                if intent == Intent.MODIFY_USER:
                    if not conversation.user_id:
                        raise ValueError("You must be logged in to modify your account.")
                    
                    # Add user_id to params
                    params['user_id'] = conversation.user_id
                    
                    user_data = params.get('user_data', {})
                    if not user_data:
                        # here we should ask for the fields to update
                        bot_message.content = "What fields would you like to update?"
                        self.db.add(bot_message)
                        self.db.commit()
                        self.db.refresh(bot_message)
                        self.db.refresh(user_message)
                        return user_message, bot_message
                    
                    # Create UserModify object with the fields to update
                    user_data = UserModify(**user_data)
                    
                    # Call the user service to update the user
                    try:
                        updated_user = self.user_service.modify_user(conversation.user_id, user_data)
                        updated_fields = [field for field in ['name', 'email', 'password'] 
                                       if getattr(user_data, field) is not None]
                        
                        function_result = {"updated_fields": updated_fields}
                        status = FunctionCallStatus.COMPLETED if updated_fields else FunctionCallStatus.FAILED

                        # Update bot message with multi-field response
                        if status == FunctionCallStatus.COMPLETED:
                            field_messages = []
                            for field in updated_fields:
                                if field == 'name':
                                    field_messages.append(f"name to {user_data.name}")
                                elif field == 'email':
                                    field_messages.append(f"email to {user_data.email}")
                                elif field == 'password':
                                    field_messages.append("password")
                            
                            if len(field_messages) == 1:
                                bot_message.content = f"I've updated your {field_messages[0]}."
                            elif len(field_messages) == 2:
                                bot_message.content = f"I've updated your {field_messages[0]} and {field_messages[1]}."
                            else:
                                last_field = field_messages.pop()
                                bot_message.content = f"I've updated your {', '.join(field_messages)}, and {last_field}."
                            
                            bot_message.content += " Is there anything else you'd like to update?"
                            self.db.add(bot_message)
                    except UserAlreadyExistsError as e:
                        raise ValueError(str(e))
                else:
                    # Handle other function calls here
                    function_result = None
                    status = FunctionCallStatus.PENDING
                # create function call object and add to db
                function_call = FunctionCall(
                    message_id=user_message.id,
                    function_name=function_info["name"],
                    parameters=params,
                    result=function_result,
                    status=status,
                    completed_at=datetime.utcnow() if status == FunctionCallStatus.COMPLETED else None
                )
                self.db.add(function_call)
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
        
        # Generate bot response
        response_content = self.generate_response(intent, message_data.content, params, current_context, bot_message.content)
        bot_message.content = response_content

        self.db.add(bot_message)
        self.db.commit()
        self.db.refresh(user_message)
        self.db.refresh(bot_message)
        return user_message, bot_message

    def generate_response(self, intent: Intent, message: str, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None, msg_content: str = None) -> str:
        if msg_content:
            return msg_content
        if intent == Intent.GREETING:
            # Clear context on greeting
            if self.conversation:
                self.clear_conversation_context(self.conversation)
            return ("Hello! I'm your shopping assistant. I can help you browse products, check orders, "
                   "and manage your account. What would you like to do?")
        
        elif intent == Intent.PRODUCT_SEARCH:
            category = params.get('category', 'products')
            max_price = params.get('max_price')
            
            # If we're in product search context but missing parameters
            if context and context.get('current_intent') == Intent.PRODUCT_SEARCH.value:
                if not category and not max_price:
                    return ("What kind of products are you looking for? You can specify:\n"
                           "- A category (e.g., electronics, clothing, books)\n"
                           "- A price range (e.g., under $100)\n"
                           "- Or both!")
            
            if max_price:
                return f"I'll help you find {category} under ${max_price}. Let me search our catalog..."
            return f"I'll help you find {category}. Let me search our catalog..."
        
        elif intent == Intent.ORDER_STATUS:
            order_id = params.get('order_id')
            
            # If we're in order status context but missing order ID
            if not order_id and context and context.get('current_intent') == Intent.ORDER_STATUS.value:
                return "Could you please provide your order number? It should be a number like #1234"
            
            if order_id:
                return f"Let me check the status of order #{order_id}..."
            return "Could you please provide your order number?"
        
        elif intent == Intent.HELP:
            return ("I'm here to help! You can:\n"
                   "- Browse products by saying 'show me products' or 'search for electronics'\n"
                   "- Check order status by saying 'track my order #123'\n"
                   "- Update your profile by saying things like:\n"
                   "  • 'change my name to John'\n"
                   "  • 'update my email to new@example.com'\n"
                   "  • 'change my password'\n"
                   "What would you like to do?")
        
        elif intent == Intent.MODIFY_USER:
            if not self.conversation.user_id:
                return "You need to be logged in to modify your account. Would you like to log in first?"
            
            fields = params.get('fields', [])
            
            # If no fields specified but we're in MODIFY_USER context
            if not fields and context and context.get('current_intent') == Intent.MODIFY_USER.value:
                return ("What would you like to update? You can say things like:\n"
                       "- Change my name to John\n"
                       "- Update my email to new@example.com\n"
                       "- Change my password to NewPass123!")
            
            # Check for missing values for specified fields
            for field in fields:
                if field == 'name' and not params.get('new_name'):
                    return "What would you like your new name to be?"
                elif field == 'email' and not params.get('new_email'):
                    return "What email address would you like to use?"
                elif field == 'password' and not params.get('new_password'):
                    return "Please provide your new password. Remember it must include uppercase, lowercase, numbers, and special characters."
            
            # If we have fields but no values yet, maintain context
            if fields and not any(params.get(f'new_{field}') for field in fields):
                return f"What would you like your new {fields[0]} to be?"
            
            return f"I'll update your profile information..."
        
        return "I'm not sure I understand. Could you please rephrase that or ask for help?"

    def end_conversation(self, conversation_id: int) -> Conversation:
        conversation = self.get_conversation(conversation_id)
        conversation.active = False
        conversation.ended_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(conversation)
        return conversation 