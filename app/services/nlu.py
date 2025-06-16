from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import Tuple, Dict, List
import numpy as np
from app.models.conversations import Intent
import logging
import spacy
import re
from typing import Optional
from datetime import datetime

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
                "search for laptops",
                "search for devices",
                "find laptops",
                "looking for a laptop",
                "show me laptops",
                "browse laptop catalog",
                "search for accessories",
                "find laptop accessories",
                "need a charger",
                "looking for mouse",
                "show me laptop bags",
                "find laptop cases",
                "need keyboard",
                "search for monitors",
                "find docking station",
                "looking for webcam",
                "show me headphones",
                "find laptop stand",
                "need cooling pad",
                "search for RAM",
                "find SSD",
                "looking for external drive",
                "show me USB hub",
                "gaming laptops",
                "business laptops",
                "student laptops"
            ],
            Intent.ORDER_STATUS: [
                "check order status",
                "track order",
                "where is my order",
                "delivery status",
                "shipping status",
                "when will my laptop arrive",
                "track my delivery",
                "order tracking",
                "package status"
            ],
            Intent.HELP: [
                "need help",
                "support needed",
                "assistance required",
                "how to choose laptop",
                "laptop buying guide",
                "compare laptops",
                "which laptop should I buy",
                "laptop specifications help",
                "technical support",
                "warranty information",
                "return policy"
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
            # First, try to extract category as it's more important for initial filtering
            category = self._extract_category(text, doc)
            if category:
                params['category'] = category

            # Extract general search query, but be smarter about it
            query_terms = []
            skip_next = False  # To handle multi-word terms
            
            for i, token in enumerate(doc):
                if skip_next:
                    skip_next = False
                    continue
                
                # Skip common stop words and basic verbs
                if token.text.lower() in ['want', 'new', 'need', 'looking', 'for', 'me', 'show', 'find', 'search', 'get', 'buy']:
                    continue
                
                # Check for important adjective + noun combinations
                if token.pos_ == 'ADJ' and i < len(doc) - 1 and doc[i + 1].pos_ == 'NOUN':
                    combined_term = f"{token.text} {doc[i + 1].text}"
                    query_terms.append(combined_term)
                    skip_next = True
                # Handle individual important terms
                elif token.pos_ in ['NOUN', 'PROPN'] or (token.pos_ == 'ADJ' and token.text.lower() in ['new', 'used', 'gaming', 'business']):
                    query_terms.append(token.text)
            
            # If we found any terms, combine them into a query
            if query_terms:
                # If we have a category, we might want to be more selective about the query
                if category:
                    # Filter out the category name from query terms if it's already captured
                    query_terms = [term for term in query_terms 
                                 if term.lower() != category.lower() and 
                                 term.lower() not in self._get_category_keywords(category)]
                
                if query_terms:  # Only add query if we have terms after filtering
                    params['query'] = ' '.join(query_terms).strip()

            # Extract price ranges using both NER and pattern matching
            min_price, max_price = self._extract_price_range(text, doc)
            if min_price is not None:
                params['min_price'] = min_price
            if max_price is not None:
                params['max_price'] = max_price
            
            # Extract brand
            brand = self._extract_brand(text, doc)
            if brand:
                params['brand'] = brand

            # Extract stock status
            in_stock = self._extract_stock_status(text)
            if in_stock is not None:
                params['in_stock'] = in_stock

            # Extract pagination parameters
            page, page_size = self._extract_pagination(text)
            if page is not None:
                params['page'] = page
            if page_size is not None:
                params['page_size'] = page_size

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

    def _extract_price_range(self, text: str, doc: spacy.tokens.Doc) -> Tuple[Optional[float], Optional[float]]:
        """Extract minimum and maximum price from text."""
        min_price = None
        max_price = None
        
        # Pattern matching for price ranges
        between_pattern = re.compile(r'between\s*\$?\s*(\d+(?:\.\d{2})?)\s*(?:and|to)\s*\$?\s*(\d+(?:\.\d{2})?)', re.IGNORECASE)
        under_pattern = re.compile(r'(?:under|below|less than)\s*\$?\s*(\d+(?:\.\d{2})?)', re.IGNORECASE)
        over_pattern = re.compile(r'(?:over|above|more than)\s*\$?\s*(\d+(?:\.\d{2})?)', re.IGNORECASE)
        
        # Check for "between X and Y" pattern
        between_match = between_pattern.search(text)
        if between_match:
            min_price = float(between_match.group(1))
            max_price = float(between_match.group(2))
            return min_price, max_price
        
        # Check for "under X" pattern
        under_match = under_pattern.search(text)
        if under_match:
            max_price = float(under_match.group(1))
        
        # Check for "over X" pattern
        over_match = over_pattern.search(text)
        if over_match:
            min_price = float(over_match.group(1))
        
        # Extract prices from NER as fallback
        if not min_price and not max_price:
            for ent in doc.ents:
                if ent.label_ == 'MONEY':
                    price_text = re.sub(r'[^\d.]', '', ent.text)
                    try:
                        price = float(price_text)
                        # If we find a single price, assume it's a maximum
                        max_price = price
                    except ValueError:
                        continue
        
        return min_price, max_price

    def _extract_brand(self, text: str, doc: spacy.tokens.Doc) -> Optional[str]:
        """Extract brand from text."""
        BRANDS = {
            'dell': 'Dell',
            'acer': 'Acer',
            'asus': 'Asus',
            'hp': 'HP',
            'lenovo': 'Lenovo',
            'apple': 'Apple',
            'microsoft': 'Microsoft'
        }
        
        # Look for brand mentions in text
        text_lower = text.lower()
        for brand_lower, brand_proper in BRANDS.items():
            if brand_lower in text_lower:
                return brand_proper
        
        # Look for brand entities in NER
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                brand_lower = ent.text.lower()
                if brand_lower in BRANDS:
                    return BRANDS[brand_lower]
        
        return None

    def _extract_stock_status(self, text: str) -> Optional[bool]:
        """Extract stock status from text."""
        text_lower = text.lower()
        
        # Check for in stock indicators
        in_stock_patterns = [
            r'in stock',
            r'available',
            r'in store',
            r'ready to ship'
        ]
        
        # Check for out of stock indicators
        out_of_stock_patterns = [
            r'out of stock',
            r'unavailable',
            r'include.*out of stock',
            r'show.*all.*items'
        ]
        
        for pattern in in_stock_patterns:
            if re.search(pattern, text_lower):
                return True
                
        for pattern in out_of_stock_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return None

    def _extract_pagination(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract pagination parameters from text."""
        text_lower = text.lower()
        page = None
        page_size = None
        
        # Extract page number
        page_pattern = re.compile(r'page\s*(\d+)', re.IGNORECASE)
        page_match = page_pattern.search(text_lower)
        if page_match:
            try:
                page = int(page_match.group(1))
            except ValueError:
                pass
        
        # Extract page size
        size_patterns = [
            r'show\s*(\d+)\s*(?:items?|products?|results?)',
            r'(\d+)\s*(?:items?|products?|results?)\s*per\s*page'
        ]
        
        for pattern in size_patterns:
            size_match = re.search(pattern, text_lower)
            if size_match:
                try:
                    page_size = int(size_match.group(1))
                    break
                except ValueError:
                    continue
        
        return page, page_size

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
        # Predefined categories for laptop store
        categories = {
            'laptops': [
                'laptop', 'notebooks', 'macbook', 'chromebook',
                'gaming laptop', 'business laptop', 'student laptop',
                'ultrabook', 'workstation', 'Dell', 'Acer', "Asus"
            ],
            'accessories': [
                'accessory', 'mouse', 'keyboard', 'charger',
                'adapter', 'cable', 'bag', 'case', 'sleeve',
                'stand', 'cooling pad', 'webcam', 'headphones',
                'speakers', 'microphone', 'dock', 'hub'
            ],
            'storage': [
                'storage', 'ssd', 'hard drive', 'hdd', 'external drive',
                'flash drive', 'usb drive', 'memory card'
            ],
            'memory': [
                'ram', 'memory', 'memory upgrade', 'ddr4', 'ddr5'
            ],
            'displays': [
                'monitor', 'display', 'screen', 'external monitor',
                'portable monitor', 'hdmi', 'displayport'
            ],
            'networking': [
                'wifi', 'ethernet', 'network card', 'bluetooth adapter',
                'router', 'network cable'
            ]
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

    def _get_category_keywords(self, category: str) -> List[str]:
        """Get keywords associated with a category to avoid redundant query terms."""
        category_keywords = {
            'laptops': ['laptop', 'laptops', 'notebook', 'notebooks', 'computer', 'computers'],
            'accessories': ['accessory', 'accessories', 'peripheral', 'peripherals'],
            'displays': ['display', 'displays', 'monitor', 'monitors', 'screen', 'screens'],
            'storage': ['storage', 'drive', 'drives', 'ssd', 'hdd', 'disk', 'disks'],
            'memory': ['memory', 'ram', 'ddr', 'dimm'],
            'networking': ['network', 'networking', 'wifi', 'ethernet']
        }
        return category_keywords.get(category.lower(), []) 