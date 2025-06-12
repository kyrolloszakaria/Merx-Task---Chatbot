from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from app.services.nlu import NLUService

class Message(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    response: str
    function_call: Optional[Dict[str, Any]] = None
    detected_intent: Optional[str] = None

class Chatbot:
    def __init__(self):
        self.conversation_history: List[Message] = []
        self.available_functions: Dict[str, Any] = {}
        self.nlu = NLUService()

    async def process_message(self, message: str) -> ChatResponse:
        """
        Process incoming message and return appropriate response
        """
        # Add user message to history
        self.conversation_history.append(Message(role="user", content=message))
        
        # TODO: Implement intent detection
        intent = self._detect_intent(message)
        
        # TODO: Implement function calling if needed
        function_call = self._handle_function_call(intent, message)
        
        # Generate response
        response = self._generate_response(message, intent)
        
        # Add response to history
        self.conversation_history.append(Message(role="assistant", content=response))
        
        return ChatResponse(
            response=response,
            function_call=function_call,
            detected_intent=intent
        )

    def _detect_intent(self, message: str) -> Optional[str]:
        """
        Detect the intent of the user message using the NLUService.
        """
        intent, confidence = self.nlu.detect_intent(message)
        if intent is not None and confidence >= 0.5:
            return intent.value  # Return the string value of the Enum
        return None

    def _handle_function_call(self, intent: Optional[str], message: str) -> Optional[Dict[str, Any]]:
        """
        Handle function calling based on detected intent
        """
        # TODO: Implement function calling logic
        return None

    def _generate_response(self, message: str, intent: Optional[str]) -> str:
        """
        Generate appropriate response based on message and intent
        """
        # TODO: Implement proper response generation
        return f"I received your message: {message}"

    def register_function(self, name: str, func: Any) -> None:
        """
        Register a new function that can be called by the chatbot
        """
        self.available_functions[name] = func 