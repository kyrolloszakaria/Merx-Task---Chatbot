from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import Tuple, Dict, List
import numpy as np
from app.models.conversations import Intent
import logging
import spacy
import re
from typing import Optional

_classifier = None
_nlp = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1",
            device=-1
        )
    return _classifier

def get_spacy_model():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def warmup_nlu():
    # Warmup transformer model
    classifier = get_classifier()
    classifier("warmup", candidate_labels=["test"])
    
    # Warmup spaCy model
    nlp = get_spacy_model()
    nlp("Warmup text to ensure the model is loaded")

class NLUService:
    def __init__(self):
        self.classifier = get_classifier()
        self.nlp = get_spacy_model()
        
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
        
        # Lowered threshold to 0.2
        if confidence < 0.2:
            # Fallback: simple keyword match
            for label, intent in label_to_intent.items():
                if label in text.lower():
                    return intent, 0.2
            return Intent.UNKNOWN, confidence
            
        # Map the label back to our Intent enum
        detected_intent = label_to_intent[best_label]
        
        return detected_intent, confidence

    def extract_parameters(self, text: str, intent: Intent) -> Dict:
        """
        Extract relevant parameters from the text using NER and pattern matching.
        Uses spaCy for named entity recognition and custom patterns for specific fields.
        """
        # Initialize logger
        logger = logging.getLogger(__name__)
        
        # Initialize params dictionary
        params = {}
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        if intent == Intent.PRODUCT_SEARCH:
            # Extract price using both NER and pattern matching
            price = self._extract_price(text, doc)
            if price is not None:
                params['max_price'] = price
            
            # Extract category using both NER and custom matching
            category = self._extract_category(text, doc)
            if category:
                params['category'] = category
        
        elif intent == Intent.ORDER_STATUS:
            # Extract order number using pattern matching and NER
            order_id = self._extract_order_id(text, doc)
            if order_id:
                params['order_id'] = order_id
        
        elif intent == Intent.MODIFY_USER:
            # Extract user data using combined NER and pattern matching
            user_data = self._extract_user_data(text, doc)
            if user_data:
                params['user_data'] = user_data
        
        return params

    def _extract_price(self, text: str, doc: spacy.tokens.Doc) -> Optional[float]:
        """Extract price from text using both NER and pattern matching."""
        # Try NER first
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                # Extract number from money entity
                amount = re.findall(r'\d+\.?\d*', ent.text)
                if amount:
                    return float(amount[0])
        
        # Fallback to pattern matching
        price_patterns = [
            r'\$\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*dollars',
            r'under\s*\$?\s*(\d+\.?\d*)',
            r'below\s*\$?\s*(\d+\.?\d*)',
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return None

    def _extract_category(self, text: str, doc: spacy.tokens.Doc) -> Optional[str]:
        """Extract product category using NER and custom matching."""
        # Predefined categories
        categories = {
            'electronics': ['electronics', 'gadgets', 'devices', 'phones', 'computers'],
            'clothing': ['clothing', 'clothes', 'apparel', 'fashion', 'wear'],
            'books': ['books', 'novels', 'textbooks', 'magazines'],
            'food': ['food', 'groceries', 'snacks', 'drinks'],
            'furniture': ['furniture', 'chairs', 'tables', 'sofas']
        }
        
        # Check NER results
        for ent in doc.ents:
            if ent.label_ == "PRODUCT":
                # Map entity to category
                for category, keywords in categories.items():
                    if any(keyword in ent.text.lower() for keyword in keywords):
                        return category
        
        # Fallback to keyword matching
        text_lower = text.lower()
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return None

    def _extract_order_id(self, text: str, doc: spacy.tokens.Doc) -> Optional[int]:
        """Extract order ID using pattern matching and NER."""
        # Try pattern matching first
        patterns = [
            r'order\s*#?\s*(\d{4,})',
            r'#\s*(\d{4,})',
            r'order\s*number\s*(\d{4,})',
            r'order\s*id\s*(\d{4,})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Try NER as fallback
        for ent in doc.ents:
            if ent.label_ == "CARDINAL":
                number = re.findall(r'\d{4,}', ent.text)
                if number:
                    return int(number[0])
        
        return None

    def _extract_user_data(self, text: str, doc: spacy.tokens.Doc) -> Dict:
        """Extract user data using NER and pattern matching."""
        user_data = {}
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            user_data['email'] = email_match.group()
            logger = logging.getLogger(__name__)
            logger.info(f"Email found: {user_data['email']}")
        
        # Extract name using NER
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                user_data['name'] = ent.text
                break
        
        # Fallback to pattern matching for name
        if 'name' not in user_data:
            name_patterns = [
                r'name (?:to|as) (\w+)',
                r'name (?:should be|will be) (\w+)',
                r'change (?:my )?name to (\w+)'
            ]
            for pattern in name_patterns:
                name_match = re.search(pattern, text.lower())
                if name_match:
                    user_data['name'] = name_match.group(1)
                    break
        
        # Extract password (pattern matching only for security)
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
        
        return user_data 