# CPU-Optimized Streamlit App for Emotion Detection (Transformer Only)
# Save this as streamlit_app.py and run with: streamlit run streamlit_app.py

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Import only Transformer model
from main import OptimizedTransformerEmotionClassifier

# CPU optimization
torch.set_num_threads(2)
os.environ['OMP_NUM_THREADS'] = '2'

# Page setup
st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="wide")

st.title("🎭 Text Emotion Detection (CPU Optimized)")
st.markdown("### Analyze emotions in text using DistilBERT")
st.markdown("Enter any text below and our AI will predict the dominant emotion!")

# Sidebar info
st.sidebar.header("⚙️ Model Configuration")
st.sidebar.info("🤖 Using Optimized Transformer (DistilBERT)")

# Emotion labels, colors, and emojis
EMOTION_LABELS = {
    0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'
}

EMOTION_COLORS = {
    'Sadness': '#1f77b4', 'Joy': '#ff7f0e', 'Love': '#d62728',
    'Anger': '#8c564b', 'Fear': '#9467bd', 'Surprise': '#17becf'
}

EMOTION_EMOJIS = {
    'Sadness': '😢', 'Joy': '😊', 'Love': '❤️',
    'Anger': '😠', 'Fear': '😨', 'Surprise': '😲'
}

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained DistilBERT model and tokenizer with CPU optimization"""
    try:
        model = OptimizedTransformerEmotionClassifier(num_classes=6)
        checkpoint_path = "emotion_transformer_model.ckpt"

        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['state_dict'])
                st.sidebar.success("✅ Loaded trained Transformer model")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Using untrained Transformer model: {str(e)}")
        else:
            st.sidebar.warning("⚠️ No trained Transformer model found, using untrained model")

        tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased', use_fast=True
        )
        return model, tokenizer

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you have run the training script first: `python quick_train.py`")
        return None, None


def predict_emotion(model, text, tokenizer):
    """Predict emotion using Transformer"""
    if not text.strip():
        return None
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        logits = model(input_ids, attention_mask)

        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        return {
            'predicted_emotion': EMOTION_LABELS[predicted_class],
            'confidence': confidence,
            'probabilities': {EMOTION_LABELS[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }

# Load model
with st.spinner("Loading Transformer model..."):
    model, tokenizer = load_model_and_tokenizer()

if model is None:
    st.error("Could not load model. Please ensure you have trained models available.")
    st.stop()

# Input area
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📝 Enter Your Text")
    input_method = st.radio("Choose input method:", ["Type text", "Upload file"], horizontal=True)

    if input_method == "Type text":
        user_text = st.text_area(
            "Enter text to analyze:",
            placeholder="Type something... (e.g., 'I am so happy today!')",
            height=150
        )
    else:
        uploaded_file = st.file_uploader("Upload a text file:", type=['txt'])
        user_text = ""
        if uploaded_file:
            user_text = str(uploaded_file.read(), "utf-8")
            st.text_area("File content:", user_text, height=150, disabled=True)

with col2:
    st.subheader("🎯 Sample Texts")
    samples = [
        "I am so happy today! Everything is going perfectly! 🎉",
        "I'm worried about my exam tomorrow.",
        "You mean everything to me. I love you! 💕",
        "This is terrible! I can't believe this happened!",
        "Oh wow! What a surprise!",
        "I feel lonely and sad today."
    ]
    for i, s in enumerate(samples):
        if st.button(f"📄 Sample {i+1}", key=f"sample_{i}"):
            st.session_state.sample_text = s
            st.rerun()

if hasattr(st.session_state, 'sample_text'):
    user_text = st.session_state.sample_text
    del st.session_state.sample_text

# Prediction
if st.button("🔍 Analyze Emotion", type="primary"):
    if user_text and user_text.strip():
        with st.spinner("Analyzing emotion..."):
            results = predict_emotion(model, user_text, tokenizer)
            if results:
                st.session_state.results = results
                st.session_state.text = user_text
    else:
        st.warning("Please enter some text!")

# Results
if hasattr(st.session_state, 'results'):
    results = st.session_state.results
    analyzed_text = st.session_state.text
    emotion = results['predicted_emotion']
    conf = results['confidence']
    emoji = EMOTION_EMOJIS[emotion]

    st.markdown("---")
    st.subheader("📊 Analysis Results")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; border-radius:10px;
                    background-color:{EMOTION_COLORS[emotion]}20;
                    border:2px solid {EMOTION_COLORS[emotion]};">
            <h1 style="color:{EMOTION_COLORS[emotion]};">{emoji} {emotion}</h1>
            <h3 style="color:{EMOTION_COLORS[emotion]};">Confidence: {conf:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📈 Probability Breakdown")
    prob_df = pd.DataFrame([
        {'Emotion': e, 'Probability': p, 'Emoji': EMOTION_EMOJIS[e]}
        for e, p in results['probabilities'].items()
    ]).sort_values('Probability', ascending=True)

    fig = px.bar(prob_df, x='Probability', y='Emotion', orientation='h',
                 color='Emotion', color_discrete_map=EMOTION_COLORS,
                 title="Emotion Probability Distribution")
    fig.update_traces(texttemplate='%{x:.1%}', textposition='outside')
    fig.update_layout(showlegend=False, height=400, title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

# Sidebar summary
st.sidebar.markdown("---")
st.sidebar.success("🎯 Using DistilBERT Transformer")
st.sidebar.markdown("• High accuracy (~93%)\n• Optimized for CPU\n• Memory-efficient")

st.sidebar.markdown("---")
st.sidebar.markdown("*Built with Streamlit, PyTorch Lightning, and Transformers*")
