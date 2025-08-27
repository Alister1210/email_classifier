# classify.py
import re
import torch
import numpy as np
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class EmailClassifier:
    """
    Handles email text classification using the trained DistilBERT model
    """
    
    def __init__(self, model: AutoModelForSequenceClassification, 
                 tokenizer: AutoTokenizer, label_encoder: LabelEncoder, max_length: int = 512):
        """
        Initialize the EmailClassifier
        
        Args:
            model: Trained classification model
            tokenizer: Tokenizer for text preprocessing
            label_encoder: Label encoder for converting predictions to labels
            max_length: Maximum sequence length (default matches training script)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.device = next(model.parameters()).device
        self.max_length = max_length
        
        logger.info(f"EmailClassifier initialized on device: {self.device}")
        logger.info(f"Available labels: {self.label_encoder.classes_.tolist()}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text, matching training script
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        try:
            if not isinstance(text, str):
                text = str(text)
            text = text.lower()
            text = re.sub(r'\S+@\S+', ' ', text)  # Remove email addresses
            text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)  # Remove URLs
            text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
            return text.strip()
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return str(text)
    
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text for model input
        
        Args:
            text (str): Cleaned text to tokenize
            
        Returns:
            Dict: Tokenized inputs ready for model
        """
        try:
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            for key in encoded:
                encoded[key] = encoded[key].to(self.device)
            return encoded
        except Exception as e:
            logger.error(f"Error tokenizing text: {str(e)}")
            raise
    
    def predict_raw(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Get raw predictions from the model
        
        Args:
            text (str): Text to classify
            
        Returns:
            Tuple: (logits, probabilities)
        """
        try:
            clean_text = self.clean_text(text)
            if not clean_text.strip():
                logger.warning("Empty text after cleaning")
                num_labels = len(self.label_encoder.classes_)
                logits = np.zeros(num_labels)
                probs = np.ones(num_labels) / num_labels
                return logits, probs
            
            inputs = self.tokenize_text(clean_text)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()[0]
            probs = self._softmax(logits)
            return logits, probs
        except Exception as e:
            logger.error(f"Error in raw prediction: {str(e)}")
            raise
    
    def classify_email(self, text: str, return_all_scores: bool = False) -> Dict[str, Any]:
        """
        Classify email text and return structured results
        
        Args:
            text (str): Email text to classify
            return_all_scores (bool): Whether to return scores for all labels
            
        Returns:
            Dict: Classification results
        """
        try:
            if not text or not text.strip():
                return {
                    "predicted_label": "unknown",
                    "confidence": 0.0,
                    "error": "Empty or invalid text provided"
                }
            
            logits, probabilities = self.predict_raw(text)
            predicted_idx = np.argmax(probabilities)
            predicted_label = self.label_encoder.classes_[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            result = {
                "predicted_label": predicted_label,
                "confidence": confidence,
                "text_length": len(text),
                "cleaned_text_length": len(self.clean_text(text))
            }
            
            if return_all_scores:
                result["all_scores"] = {label: float(probabilities[idx]) 
                                      for idx, label in enumerate(self.label_encoder.classes_)}
            
            result["confidence_category"] = self._get_confidence_category(confidence)
            logger.info(f"Classification complete: {predicted_label} (confidence: {confidence:.4f})")
            return result
        except Exception as e:
            logger.error(f"Error in email classification: {str(e)}")
            return {
                "predicted_label": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply softmax to logits to get probabilities
        """
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def _get_confidence_category(self, confidence: float) -> str:
        """
        Categorize confidence level
        """
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"