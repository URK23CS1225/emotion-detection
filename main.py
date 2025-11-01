import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set number of threads for CPU optimization
torch.set_num_threads(4)  # Adjust based on your CPU cores

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):  # Reduced from 128
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Pre-tokenize all data to avoid repeated tokenization
        self._tokenize_all()
    
    def _tokenize_all(self):
        """Pre-tokenize all texts to speed up training"""
        print("Pre-tokenizing dataset...")
        self.tokenized_data = []
        for i, text in enumerate(self.texts):
            encoding = self.tokenizer(
                str(text),
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            self.tokenized_data.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(self.labels[i], dtype=torch.long)
            })
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

class OptimizedTransformerEmotionClassifier(pl.LightningModule):
    def __init__(self, num_classes=6, model_name='distilbert-base-uncased', dropout=0.1):  # Reduced dropout
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.save_hyperparameters()
        
        # Load pre-trained model
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze most of the BERT layers (only train last 2 layers)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 transformer layers
        for layer in self.bert.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Also unfreeze embeddings layer norm and pooler
        for param in self.bert.embeddings.LayerNorm.parameters():
            param.requires_grad = True
            
        self.dropout = nn.Dropout(dropout)
        # Smaller classifier with fewer parameters
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),  # Reduce hidden size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # Enable gradient checkpointing to save memory
        self.bert.gradient_checkpointing_enable()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        output = self.dropout(pooled_output)
        return self.classifier(output)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        # Separate learning rates for frozen and unfrozen parameters
        bert_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'bert' in name:
                    bert_params.append(param)
                else:
                    classifier_params.append(param)
        
        optimizer = AdamW([
            {'params': bert_params, 'lr': 1e-5},  # Lower LR for BERT
            {'params': classifier_params, 'lr': 3e-4}  # Higher LR for classifier
        ], weight_decay=0.01)
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

# Faster LSTM alternative for comparison
class FastLSTMEmotionClassifier(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, num_classes=6, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Use single layer LSTM for speed
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        embedded = self.embedding(x)
        if attention_mask is not None:
            # Apply attention mask
            embedded = embedded * attention_mask.unsqueeze(-1).float()
        
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use mean pooling instead of last hidden state
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            lstm_out = lstm_out * mask
            pooled = lstm_out.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = lstm_out.mean(dim=1)
            
        output = self.classifier(self.dropout(pooled))
        return output
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        labels = batch['label']
        outputs = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        labels = batch['label']
        outputs = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.003)

def load_emotion_dataset(max_samples=None):
    """Load and prepare the emotion dataset with optional sample limiting"""
    print("Loading emotion dataset...")
    dataset = load_dataset("emotion")
    
    # Optionally limit dataset size for faster training
    if max_samples:
        train_texts = dataset['train']['text'][:max_samples]
        train_labels = dataset['train']['label'][:max_samples]
        val_texts = dataset['validation']['text'][:max_samples//4]
        val_labels = dataset['validation']['label'][:max_samples//4]
    else:
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        val_texts = dataset['validation']['text']
        val_labels = dataset['validation']['label']
    
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    return {
        'train': (train_texts, train_labels),
        'validation': (val_texts, val_labels),
        'test': (test_texts, test_labels)
    }

def create_data_loaders(data, tokenizer, batch_size=32, num_workers=2):  # Increased batch size, added workers
    """Create PyTorch data loaders with optimizations"""
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['validation']
    
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length=64)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length=64)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False,  # Disable for CPU
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader

def create_lstm_data_loaders(data, tokenizer, batch_size=64, num_workers=2):
    """Create data loaders for LSTM model"""
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['validation']
    
    # Build vocabulary from tokenizer
    vocab_size = tokenizer.vocab_size
    
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length=64)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length=64)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, vocab_size

def train_model(model_type='transformer', max_epochs=5, batch_size=32, max_samples=None):
    """Train the emotion detection model with optimizations"""
    print(f"Training {model_type} model...")
    
    # Load data
    data = load_emotion_dataset(max_samples=max_samples)
    
    if model_type == 'transformer':
        # Use faster tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)
        model = OptimizedTransformerEmotionClassifier(num_classes=6)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(data, tokenizer, batch_size)
        
    elif model_type == 'lstm':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)
        train_loader, val_loader, vocab_size = create_lstm_data_loaders(data, tokenizer, batch_size=64)
        model = FastLSTMEmotionClassifier(vocab_size=vocab_size, num_classes=6)
    
    # Initialize trainer with CPU optimizations
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='cpu',
        log_every_n_steps=50,  # Log less frequently
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=False,  # Disable for speed
        enable_checkpointing=True,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                mode='min'
            ),
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename='best-model-{epoch:02d}-{val_loss:.2f}'
            )
        ]
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Save the model
    model_path = f"emotion_{model_type}_model.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Model saved to {model_path}")
    
    return model, tokenizer

def predict_emotion(text, model, tokenizer):
    """Predict emotion for a given text with optimizations"""
    model.eval()
    
    # Tokenize input with reduced max length
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=64,  # Reduced from 128
        return_tensors='pt'
    )
    
    # Get prediction
    with torch.no_grad():
        if hasattr(model, 'bert'):  # Transformer model
            outputs = model(encoding['input_ids'], encoding['attention_mask'])
        else:  # LSTM model
            outputs = model(encoding['input_ids'], encoding['attention_mask'])
        prediction = torch.argmax(outputs, dim=1).item()
    
    # Emotion labels
    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    return emotions[prediction]

if __name__ == "__main__":
    # Quick test with optimizations
    print("Testing optimized emotion detection...")
    try:
        # Test with smaller dataset first
        print("\n=== Testing Transformer (Optimized) ===")
        model_transformer, tokenizer = train_model(
            model_type='transformer', 
            max_epochs=3, 
            batch_size=32,
            max_samples=2000  # Use smaller dataset for testing
        )
        
        # Test prediction
        test_text = "I am so happy today!"
        emotion = predict_emotion(test_text, model_transformer, tokenizer)
        print(f"Text: '{test_text}'")
        print(f"Predicted emotion (Transformer): {emotion}")
        
        print("\n=== Testing LSTM (Fast) ===")
        model_lstm, tokenizer_lstm = train_model(
            model_type='lstm', 
            max_epochs=5, 
            batch_size=64,
            max_samples=2000
        )
        
        emotion_lstm = predict_emotion(test_text, model_lstm, tokenizer_lstm)
        print(f"Predicted emotion (LSTM): {emotion_lstm}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()