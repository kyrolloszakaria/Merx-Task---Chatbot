from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import Tuple, Dict, List
import numpy as np
from app.models.conversations import Intent

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
                "greeting", "hello", "hi", "welcome", "hey", "good morning", "good evening", "good afternoon", "say hello", "introduce myself"
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
        params = {}
        
        if intent == Intent.PRODUCT_SEARCH:
            # Extract potential price mentions
            if any(word in text.lower() for word in ['under', 'below', 'max', 'less than']):
                # Simple price extraction (could be improved with regex or NER)
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
        
        return params 