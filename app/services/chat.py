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
from app.services.products import ProductService
from app.schemas.products import ProductSearchParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.nlu = NLUService()
        self.user_service = UserService(db)
        self.product_service = ProductService(db)
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
                "description": "Search for products in the laptop store catalog",
                "parameters": {
                    "query": {
                        "type": "string?",
                        "description": "General search term for product name or description"
                    },
                    "brand": {
                        "type": "string?",
                        "description": "Brand name (e.g., Dell, Acer, Asus, HP, Lenovo)",
                        "enum": ["Dell", "Acer", "Asus", "HP", "Lenovo", "Apple", "Microsoft"]
                    },
                    "category": {
                        "type": "string?",
                        "description": "Product category",
                        "enum": ["laptops", "accessories", "displays", "storage", "memory", "networking"]
                    },
                    "max_price": {
                        "type": "float?",
                        "description": "Maximum price in dollars",
                        "minimum": 0
                    },
                    "min_price": {
                        "type": "float?",
                        "description": "Minimum price in dollars",
                        "minimum": 0
                    },
                    "in_stock": {
                        "type": "boolean?",
                        "description": "Filter for products in stock"
                    },
                    "page": {
                        "type": "integer?",
                        "description": "Page number for pagination",
                        "default": 1,
                        "minimum": 1
                    },
                    "page_size": {
                        "type": "integer?",
                        "description": "Number of items per page",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "examples": [
                    "Show me Dell laptops under $1000",
                    "Find gaming laptops in stock",
                    "Search for laptop accessories",
                    "Show me monitors under $500",
                    "Find SSD storage devices",
                    "Search for laptop chargers",
                    "Show me laptops between $500 and $1000",
                    "Find Dell XPS laptops"
                ]
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
                elif intent == Intent.PRODUCT_SEARCH:
                    # Call product service to search products
                    try:
                        # Extract pagination parameters
                        page = params.pop('page', 1)
                        page_size = params.pop('page_size', 20)
                        skip = (page - 1) * page_size
                        
                        # Create search params object
                        search_params = ProductSearchParams(**params)
                        
                        # Perform search
                        products, total = self.product_service.search_products(
                            search_params=search_params,
                            skip=skip,
                            limit=page_size
                        )
                        
                        # Format the search results for the response
                        function_result = {
                            "total": total,
                            "page": page,
                            "page_size": page_size,
                            "products": [
                                {
                                    "id": product.id,
                                    "name": product.name,
                                    "price": product.price,
                                    "category": product.category,
                                    "in_stock": product.stock > 0
                                }
                                for product in products
                            ]
                        }
                        status = FunctionCallStatus.COMPLETED
                        
                        # Generate response based on search results
                        if total == 0:
                            bot_message.content = self._generate_no_results_response(params)
                        else:
                            bot_message.content = self._generate_search_results_response(function_result, params)
                            
                    except Exception as e:
                        logger.error(f"Error during product search: {str(e)}")
                        function_result = {"error": str(e)}
                        status = FunctionCallStatus.FAILED
                        bot_message.content = f"I apologize, but I encountered an error while searching for products: {str(e)}"
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
            return ("Hello! I'm your laptop store assistant. I can help you find laptops and accessories, "
                   "check your orders, and manage your account. What are you looking for today?")
        
        elif intent == Intent.PRODUCT_SEARCH:
            # Extract all search parameters
            category = params.get('category')
            brand = params.get('brand')
            query = params.get('query')
            max_price = params.get('max_price')
            min_price = params.get('min_price')
            in_stock = params.get('in_stock')
            
            # If we're in product search context but missing key parameters
            if context and context.get('current_intent') == Intent.PRODUCT_SEARCH.value:
                if not any([category, brand, query, max_price, min_price]):
                    return ("I can help you find the perfect laptop or accessory! You can specify:\n"
                           "- A category (e.g., laptops, accessories, displays, storage)\n"
                           "- A brand (e.g., Dell, Acer, Asus, HP)\n"
                           "- A price range (e.g., under $1000 or between $500 and $1000)\n"
                           "- Or combine them (e.g., 'Dell laptops under $800')\n"
                           "What are you interested in?")
            
            # Build a natural response based on the search parameters
            response_parts = []
            
            # Add category/brand specific part
            if category and brand:
                response_parts.append(f"{brand} {category}")
            elif brand:
                response_parts.append(f"{brand} products")
            elif category:
                response_parts.append(f"{category}")
            elif query:
                response_parts.append(f"products matching '{query}'")
            
            # Add price range part
            if min_price and max_price:
                response_parts.append(f"between ${min_price} and ${max_price}")
            elif max_price:
                response_parts.append(f"under ${max_price}")
            elif min_price:
                response_parts.append(f"over ${min_price}")
            
            # Add stock status
            if in_stock is True:
                response_parts.append("that are in stock")
            elif in_stock is False:
                response_parts.append("including out of stock items")
            
            if response_parts:
                search_criteria = ", ".join(response_parts[:-1])
                if len(response_parts) > 1:
                    search_criteria += f" and {response_parts[-1]}"
                else:
                    search_criteria = response_parts[0]
                return f"I'll help you find {search_criteria}. Let me search our catalog..."
            
            return "I'll search our catalog. Please let me know if you want to filter by brand, price, or category."
        
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
                   "- Browse our laptop store by saying things like:\n"
                   "  • 'Show me Dell laptops'\n"
                   "  • 'Find gaming laptops under $1500'\n"
                   "  • 'Search for laptop chargers'\n"
                   "  • 'Show me monitors in stock'\n"
                   "- Check your order status by saying 'track my order #123'\n"
                   "- Update your profile by saying:\n"
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

    def _generate_no_results_response(self, params: Dict[str, Any]) -> str:
        """Generate a response when no products are found."""
        response_parts = []
        
        # Add specific reasons why no results were found
        if params.get('brand'):
            response_parts.append(f"no {params['brand']} products")
        if params.get('category'):
            response_parts.append(f"in the {params['category']} category")
        if params.get('min_price') and params.get('max_price'):
            response_parts.append(f"between ${params['min_price']} and ${params['max_price']}")
        elif params.get('max_price'):
            response_parts.append(f"under ${params['max_price']}")
        elif params.get('min_price'):
            response_parts.append(f"over ${params['min_price']}")
        if params.get('in_stock') is True:
            response_parts.append("currently in stock")
        
        if response_parts:
            criteria = ", ".join(response_parts[:-1])
            if len(response_parts) > 1:
                criteria += f" and {response_parts[-1]}"
            else:
                criteria = response_parts[0]
            message = f"I couldn't find any products matching your criteria: {criteria}."
        else:
            message = "I couldn't find any products matching your search criteria."
        
        # Add suggestions
        message += "\n\nYou could try:"
        suggestions = []
        if params.get('min_price') or params.get('max_price'):
            suggestions.append("• Adjusting your price range")
        if params.get('brand'):
            suggestions.append(f"• Looking for a different brand than {params['brand']}")
        if params.get('in_stock') is True:
            suggestions.append("• Including out of stock items")
        if params.get('category'):
            suggestions.append("• Browsing a different category")
        suggestions.append("• Using more general search terms")
        
        return message + "\n" + "\n".join(suggestions)

    def _generate_search_results_response(self, results: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Generate a response for successful product search."""
        total_results = results["total"]
        products = results["products"]
        current_page = results["page"]
        page_size = results["page_size"]
        
        # Start with the number of results
        if total_results == 1:
            message = "I found 1 product"
        else:
            message = f"I found {total_results} products"
        
        # Add search criteria used
        criteria_parts = []
        if params.get('brand'):
            criteria_parts.append(f"from {params['brand']}")
        if params.get('category'):
            criteria_parts.append(f"in {params['category']}")
        if params.get('min_price') and params.get('max_price'):
            criteria_parts.append(f"between ${params['min_price']} and ${params['max_price']}")
        elif params.get('max_price'):
            criteria_parts.append(f"under ${params['max_price']}")
        elif params.get('min_price'):
            criteria_parts.append(f"over ${params['min_price']}")
        if params.get('in_stock') is True:
            criteria_parts.append("in stock")
        
        if criteria_parts:
            criteria = ", ".join(criteria_parts[:-1])
            if len(criteria_parts) > 1:
                criteria += f" and {criteria_parts[-1]}"
            else:
                criteria = criteria_parts[0]
            message += f" {criteria}"
        
        message += ":\n\n"
        
        # Add product details
        for i, product in enumerate(products, 1):
            message += (f"{i}. {product['name']} - ${product['price']:.2f} "
                       f"({'In Stock' if product['in_stock'] else 'Out of Stock'})\n")
        
        # Add pagination info if necessary
        total_pages = (total_results + page_size - 1) // page_size
        if total_pages > 1:
            message += f"\nShowing page {current_page} of {total_pages}. "
            if current_page < total_pages:
                message += "You can ask for the next page to see more results."
        
        return message 