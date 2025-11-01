# api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import torch
import asyncio
import time
import logging
from datetime import datetime
import uvicorn
import json
import os
from pathlib import Path

# Import your model classes
# from your_models import OptimizedTransformerEmotionClassifier, FastLSTMEmotionClassifier
# from transformers import DistilBertTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion Detection API",
    description="Real-time emotion detection from text using DistilBERT and LSTM models",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Input text for emotion detection")
    model_type: str = Field(default="transformer", description="Model to use: 'transformer' or 'lstm'")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts for batch processing")
    model_type: str = Field(default="transformer", description="Model to use: 'transformer' or 'lstm'")

class EmotionPrediction(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: Dict[str, float]
    processing_time_ms: float
    model_used: str

class BatchEmotionPrediction(BaseModel):
    predictions: List[EmotionPrediction]
    total_processing_time_ms: float
    average_time_per_text_ms: float

class ModelStats(BaseModel):
    model_name: str
    total_predictions: int
    average_processing_time_ms: float
    uptime_seconds: float

# Global variables for model management
models = {}
tokenizers = {}
model_stats = {
    "transformer": {"total_predictions": 0, "total_time": 0},
    "lstm": {"total_predictions": 0, "total_time": 0}
}
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
start_time = time.time()

async def load_models():
    """Load models asynchronously on startup"""
    try:
        logger.info("Loading models...")
        
        # Load DistilBERT model and tokenizer
        # tokenizers["transformer"] = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # models["transformer"] = OptimizedTransformerEmotionClassifier.load_from_checkpoint("path/to/transformer/checkpoint.ckpt")
        # models["transformer"].eval()
        
        # Load LSTM model
        # models["lstm"] = FastLSTMEmotionClassifier.load_from_checkpoint("path/to/lstm/checkpoint.ckpt")
        # models["lstm"].eval()

    except Exception as e:
        logger.error(f"Error loading models: {e}")
       