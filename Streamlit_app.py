# CPU-Optimized Streamlit App for Emotion Detection
# Save this as streamlit_app.py and run with: streamlit run streamlit_app.py

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import plotly.express as px
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Import your optimized models with correct names
from main import OptimizedTransformerEmotionClassifier, FastLSTMEmotionClassifier

# Set CPU optimizations
torch.set_num_threads(2)  # Reduce for Streamlit to prevent blocking
os.environ['OMP_NUM_THREADS'] = '2'

# Configure page
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide"
)

# Title and description
st.title("🎭 Text Emotion Detection (CPU Optimized)")
st.markdown("### Analyze emotions in text using CPU-optimized AI models")
st.markdown("Enter any text below and our AI will predict the dominant emotion!")

# Sidebar for model configuration
st.sidebar.header("⚙️ Model Configuration")
model_type = st.sidebar.selectbox(
    "Choose Model Type:",
    ["Optimized Transformer (DistilBERT)", "Fast LSTM"],
    help="Transformer models are more accurate, LSTM is faster"
)

# Emotion labels and colors
EMOTION_LABELS = {
    0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'
}

EMOTION_COLORS = {
    'Sadness': '#1f77b4',   # Blue
    'Joy': '#ff7f0e',       # Orange
    'Love': '#d62728',      # Red
    'Anger': '#8c564b',     # Brown
    'Fear': '#9467bd',      # Purple
    'Surprise': '#17becf'   # Cyan
}

EMOTION_EMOJIS = {
    'Sadness': '😢',
    'Joy': '😊',
    'Love': '❤️',
    'Anger': '😠',
    'Fear': '😨',
    'Surprise': '😲'
}

@st.cache_resource
def load_model_and_tokenizer(model_type):
    """Load the trained model and tokenizer with CPU optimizations"""
    try:
        if "Transformer" in model_type:
            # Load optimized transformer model
            model = OptimizedTransformerEmotionClassifier(num_classes=6)
            
            # Try to load from checkpoint
            checkpoint_path = "emotion_transformer_model.ckpt"
            if os.path.exists(checkpoint_path):
                try:
                    # Load Lightning checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    model.load_state_dict(checkpoint['state_dict'])
                    st.sidebar.success("✅ Loaded trained Transformer model")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Using untrained Transformer model: {str(e)}")
            else:
                st.sidebar.warning("⚠️ No trained Transformer model found, using untrained model")
            
            # Load tokenizer with CPU optimization
            tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased', 
                use_fast=True  # Use fast tokenizer for better CPU performance
            )
            return model, tokenizer, "transformer"
            
        else:
            # Load LSTM model
            # We need to get vocab_size from tokenizer first
            tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased', 
                use_fast=True
            )
            vocab_size = tokenizer.vocab_size
            
            model = FastLSTMEmotionClassifier(
                vocab_size=vocab_size, 
                embedding_dim=64,  # Match your training config
                hidden_dim=64,
                num_classes=6
            )
            
            # Try to load from checkpoint
            checkpoint_path = "emotion_lstm_model.ckpt"
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    model.load_state_dict(checkpoint['state_dict'])
                    st.sidebar.success("✅ Loaded trained LSTM model")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Using untrained LSTM model: {str(e)}")
            else:
                st.sidebar.warning("⚠️ No trained LSTM model found, using untrained model")
            
            return model, tokenizer, "lstm"
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you have run the training script first: `python quick_train.py`")
        return None, None, None

@st.cache_data
def predict_emotion(model, user_text, _tokenizer, loaded_model_type):
    # Your existing function code here
    pass
def predict_emotion(_model, text, tokenizer, model_type):
    """Predict emotion for input text with CPU optimizations"""
    if not text.strip():
        return None
    
    # Ensure model is in eval mode
    _model.eval()
    
    # CPU optimization: disable gradient computation
    with torch.no_grad():
        if model_type == "transformer":
            # Tokenize with reduced max_length for speed (matching training)
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=64,  # Reduced from 128 to match training optimization
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            # Get prediction
            logits = _model(input_ids, attention_mask)
            
        else:  # LSTM
            # Tokenize for LSTM
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=64,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids']
            attention_mask = encoding.get('attention_mask')
            
            # Get prediction
            logits = _model(input_ids, attention_mask)
        
        # Get probabilities
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Create results dictionary
        results = {
            'predicted_emotion': EMOTION_LABELS[predicted_class],
            'confidence': confidence,
            'probabilities': {EMOTION_LABELS[i]: prob.item() 
                            for i, prob in enumerate(probabilities[0])}
        }
        
        return results

# Load model with error handling
with st.spinner("Loading model (this may take a moment on first run)..."):
    model, tokenizer, loaded_model_type = load_model_and_tokenizer(model_type)

if model is None:
    st.error("Could not load model. Please ensure you have trained models available.")
    st.info("Run `python quick_train.py` first to train your models.")
    st.stop()

# Performance indicator
model_info = "🚀 Fast LSTM" if loaded_model_type == "lstm" else "🤖 Optimized Transformer"
st.sidebar.markdown(f"**Current Model:** {model_info}")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Text")
    
    # Text input options
    input_method = st.radio(
        "Choose input method:",
        ["Type text", "Upload file"],
        horizontal=True
    )
    
    if input_method == "Type text":
        user_text = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste your text here... (e.g., 'I am so excited about this new project!')",
            height=150,
            help="Enter any text and we'll predict its emotional content. Optimized for CPU performance!"
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a text file:",
            type=['txt'],
            help="Upload a .txt file to analyze its content"
        )
        
        user_text = ""
        if uploaded_file is not None:
            try:
                user_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", user_text, height=150, disabled=True)
            except Exception as e:
                st.error(f"Error reading file: {e}")

with col2:
    st.subheader("🎯 Sample Texts")
    sample_texts = [
        "I am so happy today! Everything is going perfectly! 🎉",
        "I'm really worried about the upcoming exam. I don't think I'm prepared enough.",
        "You mean everything to me. I love you so much! 💕",
        "This is absolutely terrible! I can't believe this happened!",
        "Oh wow! I never expected this to happen. What a surprise!",
        "I feel so lonely and sad today. Nothing seems to go right."
    ]
    
    st.markdown("**Click to try these examples:**")
    for i, sample in enumerate(sample_texts):
        if st.button(f"📄 Sample {i+1}", key=f"sample_{i}", help=sample[:50]+"..."):
            st.session_state.sample_text = sample
            st.rerun()

# Handle sample text selection
if hasattr(st.session_state, 'sample_text'):
    user_text = st.session_state.sample_text
    # Clear the session state to avoid persistence
    del st.session_state.sample_text

# Prediction section
analyze_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)

if analyze_button and user_text and user_text.strip():
    # Show performance info
    performance_text = "⚡ Fast prediction" if loaded_model_type == "lstm" else "🎯 High accuracy prediction"
    
    with st.spinner(f"Analyzing emotion... {performance_text}"):
        results = predict_emotion(model, user_text, tokenizer, loaded_model_type)
        
        if results:
            # Store results in session state
            st.session_state.results = results
            st.session_state.analyzed_text = user_text
            st.session_state.model_used = loaded_model_type
elif analyze_button:
    st.warning("Please enter some text to analyze!")

# Display results
if hasattr(st.session_state, 'results') and st.session_state.results:
    results = st.session_state.results
    analyzed_text = st.session_state.analyzed_text
    model_used = st.session_state.get('model_used', loaded_model_type)
    
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Show which model was used
    model_badge = "🚀 LSTM" if model_used == "lstm" else "🤖 Transformer"
    st.caption(f"Analyzed using: {model_badge}")
    
    # Main prediction display
    predicted_emotion = results['predicted_emotion']
    confidence = results['confidence']
    emoji = EMOTION_EMOJIS[predicted_emotion]
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px; 
                    background-color: {EMOTION_COLORS[predicted_emotion]}20; 
                    border: 2px solid {EMOTION_COLORS[predicted_emotion]};">
            <h1 style="color: {EMOTION_COLORS[predicted_emotion]}; margin: 0;">
                {emoji} {predicted_emotion}
            </h1>
            <h3 style="color: {EMOTION_COLORS[predicted_emotion]}; margin: 10px 0;">
                Confidence: {confidence:.1%}
            </h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed probability breakdown
    st.subheader("📈 Probability Breakdown")
    
    # Create DataFrame for plotting
    prob_df = pd.DataFrame([
        {'Emotion': emotion, 'Probability': prob, 'Emoji': EMOTION_EMOJIS[emotion]}
        for emotion, prob in results['probabilities'].items()
    ]).sort_values('Probability', ascending=True)
    
    # Horizontal bar chart
    fig = px.bar(
        prob_df, 
        x='Probability', 
        y='Emotion',
        orientation='h',
        color='Emotion',
        color_discrete_map=EMOTION_COLORS,
        title="Emotion Probability Distribution"
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Probability",
        yaxis_title="Emotion",
        title_x=0.5
    )
    
    # Add percentage labels
    fig.update_traces(
        texttemplate='%{x:.1%}',
        textposition='outside'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    with st.expander("📋 Detailed Probabilities"):
        metrics_df = pd.DataFrame([
            {
                'Emotion': f"{EMOTION_EMOJIS[emotion]} {emotion}",
                'Probability': f"{prob:.1%}",
                'Confidence Level': 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'
            }
            for emotion, prob in sorted(results['probabilities'].items(), 
                                      key=lambda x: x[1], reverse=True)
        ])
        
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Text analysis insights
    st.subheader("💡 Analysis Insights")
    
    # Generate insights based on results
    top_emotions = sorted(results['probabilities'].items(), key=lambda x: x[1], reverse=True)
    
    insights = []
    
    if confidence > 0.8:
        insights.append(f"🎯 **High Confidence**: The model is very confident that this text expresses {predicted_emotion.lower()}.")
    elif confidence > 0.5:
        insights.append(f"⚖️ **Moderate Confidence**: The text likely expresses {predicted_emotion.lower()}, but there's some uncertainty.")
    else:
        insights.append(f"❓ **Low Confidence**: The emotional content is ambiguous. Multiple emotions might be present.")
    
    if len([p for p in results['probabilities'].values() if p > 0.2]) > 2:
        insights.append("🌈 **Mixed Emotions**: This text contains multiple emotional elements.")
    
    # Identify secondary emotion
    if len(top_emotions) > 1 and top_emotions[1][1] > 0.15:
        secondary_emotion = top_emotions[1][0]
        insights.append(f"🔄 **Secondary Emotion**: There are also traces of {secondary_emotion.lower()} ({top_emotions[1][1]:.1%}).")
    
    for insight in insights:
        st.markdown(insight)
    
    # Performance info
    if model_used == "lstm":
        st.info("⚡ **Fast Processing**: LSTM model provides quick results, ideal for real-time applications.")
    else:
        st.info("🎯 **High Accuracy**: Transformer model provides state-of-the-art accuracy with CPU optimizations.")
    
    # Word-level analysis (simplified)
    st.subheader("🔤 Text Analysis")
    
    words = analyzed_text.split()
    word_count = len(words)
    char_count = len(analyzed_text)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Words", word_count)
    with col2:
        st.metric("Characters", char_count)
    with col3:
        st.metric("Dominant Emotion", predicted_emotion)
    with col4:
        st.metric("Confidence", f"{confidence:.1%}")

# Footer with additional information
st.markdown("---")
st.markdown("### ℹ️ About This App")

with st.expander("How does this work?"):
    st.markdown("""
    This CPU-optimized emotion detection system uses deep learning models to analyze text and predict emotions:
    
    **🤖 Models Available:**
    - **Optimized Transformer (DistilBERT)**: State-of-the-art language model fine-tuned for emotion classification
      - CPU optimized with frozen layers and gradient checkpointing
      - Reduced sequence length (64 tokens) for faster processing
    - **Fast LSTM**: Lightweight recurrent neural network optimized for speed
      - Single-layer LSTM with reduced dimensions
      - 5-10x faster than Transformer on CPU
    
    **📊 Emotions Detected:**
    - 😢 **Sadness**: Expressions of sorrow, disappointment, or melancholy
    - 😊 **Joy**: Happiness, excitement, and positive feelings
    - ❤️ **Love**: Affection, care, and romantic feelings
    - 😠 **Anger**: Frustration, annoyance, and hostile emotions
    - 😨 **Fear**: Anxiety, worry, and apprehension
    - 😲 **Surprise**: Amazement, shock, and unexpected reactions
    
    **🎯 Expected Performance:**
    - Optimized Transformer: ~92-95% accuracy
    - Fast LSTM: ~85-90% accuracy
    
    **⚡ CPU Optimizations:**
    - Reduced thread count for Streamlit compatibility
    - Model layer freezing and gradient checkpointing
    - Fast tokenizers and reduced sequence lengths
    - Efficient caching and prediction optimization
    
    **💡 Tips for Better Results:**
    - Use complete sentences when possible
    - Include context and descriptive words
    - Longer texts generally produce more accurate results
    - LSTM is faster for real-time use, Transformer for highest accuracy
    """)

with st.expander("Technical Details"):
    st.markdown(f"""
    **Current Configuration:**
    - Model Type: {model_type}
    - Loaded Model: {loaded_model_type}
    - Framework: PyTorch Lightning
    - Device: CPU (optimized)
    - Tokenizer: {'DistilBERT Fast Tokenizer' if loaded_model_type == 'transformer' else 'DistilBERT for LSTM'}
    - Classes: 6 emotions
    - Max Sequence Length: 64 tokens (optimized)
    - Thread Count: 2 (Streamlit optimized)
    
    **Model Architecture:**
    - {'DistilBERT (frozen) + Classification Head' if loaded_model_type == 'transformer' else 'Embedding + Single LSTM + Dense'}
    - Loss Function: Cross-Entropy Loss
    - Optimizer: {'AdamW with differential learning rates' if loaded_model_type == 'transformer' else 'Adam'}
    - Memory: {'Gradient checkpointing enabled' if loaded_model_type == 'transformer' else 'Lightweight architecture'}
    """)

# Sidebar with additional features
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Performance Info")

# Show model performance characteristics
if loaded_model_type == "lstm":
    st.sidebar.success("🚀 **Fast LSTM Model**")
    st.sidebar.markdown("""
    - **Speed**: Very Fast ⚡
    - **Memory**: Low 💾
    - **Accuracy**: Good 📊
    - **Best for**: Real-time apps
    """)
else:
    st.sidebar.info("🤖 **Optimized Transformer**")
    st.sidebar.markdown("""
    - **Speed**: Moderate ⏱️
    - **Memory**: Optimized 💾
    - **Accuracy**: Excellent 📊
    - **Best for**: High accuracy needs
    """)

if hasattr(st.session_state, 'results'):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Current Prediction")
    
    # Show current prediction stats
    results = st.session_state.results
    st.sidebar.metric(
        "Prediction", 
        results['predicted_emotion'],
        f"{results['confidence']:.1%} confidence"
    )
    
    # Show probability distribution in sidebar
    prob_data = results['probabilities']
    for emotion, prob in sorted(prob_data.items(), key=lambda x: x[1], reverse=True):
        st.sidebar.progress(prob, text=f"{EMOTION_EMOJIS[emotion]} {emotion}: {prob:.1%}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Features")
st.sidebar.markdown("""
- CPU-optimized inference
- Real-time emotion prediction
- Interactive visualizations
- Confidence scoring
- Multiple model support
- Sample text examples
- Detailed probability breakdown
- Fast tokenization
- Memory efficient processing
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*Built with Streamlit, PyTorch Lightning, and Transformers*")
st.sidebar.markdown("*Optimized for CPU performance*")