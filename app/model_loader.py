# model_loader.py
import os
import torch
import json
import pickle
import logging
from typing import Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Handles loading of the trained model, tokenizer, and label encoder
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the ModelLoader
        
        Args:
            model_path (str): Path to the directory containing the saved model
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelLoader initialized with path: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def load_components(self) -> Tuple[Any, Any, Any, int]:
        """
        Load model, tokenizer, label encoder, and max_length
        
        Returns:
            Tuple: (model, tokenizer, label_encoder, max_length)
        """
        try:
            metadata_path = os.path.join(self.model_path, "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                max_length = metadata.get("max_length", 512)
            else:
                max_length = 512
                logger.warning("No model_metadata.json found, using default max_length=512")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            model.to(self.device)
            model.eval()
            
            label_encoder_path = os.path.join(self.model_path, "label_encoder.pkl")
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
            else:
                logger.warning("No label_encoder.pkl found, creating from metadata")
                label_encoder = LabelEncoder()
                if os.path.exists(metadata_path):
                    label_encoder.classes_ = np.array(metadata["labels"])
                else:
                    raise FileNotFoundError("Cannot initialize label encoder: missing metadata and pickle")
            
            logger.info(f"Loaded components: {model.__class__.__name__}, {tokenizer.__class__.__name__}, labels={label_encoder.classes_.tolist()}")
            return model, tokenizer, label_encoder, max_length
        except Exception as e:
            logger.error(f"Error loading components: {str(e)}")
            raise