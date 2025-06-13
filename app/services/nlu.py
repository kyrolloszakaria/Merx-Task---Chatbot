from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import Tuple, Dict, List
import numpy as np
from app.models.conversations import Intent
import logging

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1",
            device=-1
        )
    return _classifier

def warmup_nlu():
    classifier = get_classifier()
    classifier("warmup", candidate_labels=["test"])

class NLUService:
    def __init__(self):
        self.classifier = get_classifier()
        
        # Expanded candidate labels for better matching
        self.intent_labels = {
            Intent.GREETING: [
                "greeting", "hello", "hi", "welcome", "hey", 
                "good morning", "good evening", "good afternoon", 
                "say hello", "introduce myself"
            ],
            Intent.PRODUCT_SEARCH: [
                "search for products",
                "find items",
                "looking for products",
                "show products",
                "browse catalog"
            ],
            Intent.ORDER_STATUS: [
                "check order status",
                "track order",
                "where is my order",
                "delivery status",
                "shipping status"
            ],
            Intent.HELP: [
                "need help",
                "support needed",
                "assistance required",
                "how to",
                "guide me"
            ],
            Intent.MODIFY_USER: [
                "update profile",
                "modify account",
                "change my details",
                "update my information",
                "edit profile",
                "change password",
                "update email",
                "change email",
                "update name",
                "change name"
            ]
        }

    def detect_intent(self, text: str) -> Tuple[Intent, float]:
        # Flatten all possible labels
        all_labels = []
        label_to_intent = {}
        
        for intent, labels in self.intent_labels.items():
            for label in labels:
                all_labels.append(label)
                label_to_intent[label] = intent
        
        # Perform zero-shot classification
        result = self.classifier(
            text,
            candidate_labels=all_labels,
            multi_label=False
        )
        
        # Get the best matching label and its score
        best_label = result['labels'][0]
        confidence = result['scores'][0]
        
        # Lowered threshold to 0.3
        if confidence < 0.3:
            # Fallback: simple keyword match
            for label, intent in label_to_intent.items():
                if label in text.lower():
                    return intent, 0.3
            return Intent.UNKNOWN, confidence
            
        # Map the label back to our Intent enum
        detected_intent = label_to_intent[best_label]
        
        return detected_intent, confidence

    def extract_parameters(self, text: str, intent: Intent) -> Dict:
        """
        Extract relevant parameters from the text based on the intent.
        This is a simple implementation that could be enhanced with
        named entity recognition (NER) models.
        """
        # Initialize logger
        logger = logging.getLogger(__name__)
        
        # Initialize params dictionary
        params = {}
        
        if intent == Intent.PRODUCT_SEARCH:
            # Extract potential price mentions
            if any(word in text.lower() for word in ['under', 'below', 'max', 'less than']):
                words = text.split()
                for i, word in enumerate(words):
                    if word.startswith('$'):
                        try:
                            params['max_price'] = float(word[1:])
                        except ValueError:
                            pass
            
            # Extract potential category mentions
            categories = ['electronics', 'clothing', 'books', 'food', 'furniture']
            for category in categories:
                if category in text.lower():
                    params['category'] = category
                    break
        
        elif intent == Intent.ORDER_STATUS:
            # Try to extract order number
            words = text.split()
            for i, word in enumerate(words):
                if word.startswith('#'):
                    try:
                        params['order_id'] = int(word[1:])
                    except ValueError:
                        pass
                elif word.isdigit() and len(word) >= 4:  # Assume it's an order number
                    params['order_id'] = int(word)
        
        elif intent == Intent.MODIFY_USER:
            # Determine what fields to update
            update_fields = {
                'name': ['name', 'username'],
                'email': ['email', 'mail', 'e-mail'],
                'password': ['password', 'pass', 'passkey']
            }
            
            # Initialize user_data structure
            user_data = {}
            
            # Extract email if present
            import re
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_match = re.search(email_pattern, text)
            if email_match:
                user_data['email'] = email_match.group()
                logger.info(f"Email found: {user_data['email']}")
            else:
                logger.info(f"Email not found")

            # Extract name if present
            name_patterns = [
                r'name (?:to|as) (\w+)',
                r'name (?:should be|will be) (\w+)'
            ]
            for pattern in name_patterns:
                name_match = re.search(pattern, text.lower())
                if name_match:
                    user_data['name'] = name_match.group(1)
                    break

            # Extract new password
            password_patterns = [
                r'(?:new )?password (?:to|as|:)\s*([^\s,\.]+)',
                r'(?:new )?pass (?:to|as|:)\s*([^\s,\.]+)',
                r'change (?:it )?to\s*([^\s,\.]+)'
            ]
            for pattern in password_patterns:
                pass_match = re.search(pattern, text, re.IGNORECASE)
                if pass_match:
                    user_data['password'] = pass_match.group(1)
                    break
            
            # Only include user_data if we found any fields to update
            if user_data:
                params['user_data'] = user_data
        
        return params 