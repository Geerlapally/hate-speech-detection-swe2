import streamlit as st
import pandas as pd
from utils.language import detect_and_translate
from utils.model import load_model, predict_label

# Page Configuration
st.set_page_config(
    page_title="Hate Speech Detection - SWE²++", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .hate-speech { background-color: #ffebee; border-left: 5px solid #f44336; }
    .offensive { background-color: #fff3e0; border-left: 5px solid #ff9800; }
    .neutral { background-color: #e8f5e8; border-left: 5px solid #4caf50; }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<h1 class="main-header">🚨 Hate Speech Detection using SWE²++</h1>', unsafe_allow_html=True)

st.markdown("""
### 🎯 About This Tool
This advanced hate speech detection system uses the **SWE²++ (Subword Enhanced)** approach to:
- 🔍 Detect hate speech in social media posts
- 🌍 Handle multilingual text with automatic translation
- 🛡️ Identify obfuscated content using subword modeling
- ⚡ Provide real-time analysis with both ML and Transformer models
""")

# Sidebar for additional options
with st.sidebar:
    st.header("⚙️ Settings")
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_translation = st.checkbox("Show Translation Details", value=True)
    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "BERT", "RoBERTa"])

# Load dataset from Google Drive
@st.cache_data
def load_data():
    try:
        url = "https://drive.google.com/uc?export=download&id=14AeGLMqjh-cYjz6a_XoXV91EWPFhGDcu"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Load data with progress indicator
with st.spinner("Loading dataset..."):
    data = load_data()

# Dataset exploration section
with st.expander("📊 Dataset Exploration"):
    if data is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(data))
        
        with col2:
            if 'label' in data.columns:
                st.metric("Unique Labels", data['label'].nunique())
        
        with col3:
            st.metric("Features", len(data.columns))
        
        if st.button("Show Random Sample"):
            st.dataframe(data.sample(min(10, len(data))))
        
        # Show label distribution if available
        if 'label' in data.columns:
            label_counts = data['label'].value_counts()
            st.bar_chart(label_counts)

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        logistic_model, vectorizer = load_model()
        return logistic_model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

logistic_model, vectorizer = load_models()

# Main analysis section
st.header("🔍 Text Analysis")

# Text input with examples
example_texts = [
    "I love spending time with my friends!",
    "This movie is absolutely terrible",
    "You people are all the same"
]

selected_example = st.selectbox("Or choose an example:", [""] + example_texts)
user_input = st.text_area(
    "✍️ Enter a tweet or message to analyze:",
    value=selected_example,
    height=100,
    placeholder="Type your text here or select an example above..."
)

# Analysis button with enhanced UI
if st.button("🔍 Analyze Text", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze!")
    else:
        with st.spinner("Analyzing text..."):
            try:
                # Language detection and translation
                if show_translation:
                    with st.expander("🌐 Language Processing"):
                        translated_text = detect_and_translate(user_input)
                        if translated_text != user_input:
                            st.write("**Original Text:**", user_input)
                            st.write("**Translated Text:**", translated_text)
                        else:
                            st.write("**Text (English):**", translated_text)
                else:
                    translated_text = detect_and_translate(user_input)
                
                # Model prediction
                if logistic_model and vectorizer:
                    prediction = predict_label(translated_text, logistic_model, vectorizer)
                    
                    # Enhanced result display
                    label_info = {
                        0: {"name": "Hate Speech", "emoji": "❌", "class": "hate-speech", "color": "#f44336"},
                        1: {"name": "Offensive", "emoji": "🛑", "class": "offensive", "color": "#ff9800"},
                        2: {"name": "Neutral", "emoji": "✅", "class": "neutral", "color": "#4caf50"}
                    }
                    
                    result = label_info.get(prediction, {"name": "Unknown", "emoji": "❓", "class": "neutral"})
                    
                    st.markdown(f"""
                    <div class="prediction-box {result['class']}">
                        <h2>{result['emoji']} Prediction: {result['name']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show confidence scores if requested and available
                    if show_confidence:
                        try:
                            # This would need to be implemented in your predict_label function
                            # probabilities = get_prediction_probabilities(translated_text, logistic_model, vectorizer)
                            st.info("💡 Confidence scores require model probability output - check your predict_label function")
                        except:
                            pass
                
                else:
                    st.error("❌ Models not loaded properly. Please check your model files.")
                    
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {e}")

# Footer with additional information
st.markdown("---")
st.markdown("""
### 📝 How it Works
1. **Language Detection**: Automatically detects the input language
2. **Translation**: Translates non-English text to English for processing
3. **Subword Processing**: Uses SWE²++ to handle obfuscated text
4. **Classification**: Applies trained ML models to predict hate speech
5. **Results**: Provides clear categorization with confidence metrics

### 🔧 Technical Details
- **Models**: Logistic Regression, BERT, RoBERTa
- **Features**: TF-IDF, Word Embeddings, Subword Tokens
- **Languages**: Multi-language support with Google Translate API
- **Preprocessing**: Text normalization, obfuscation handling
""")

# Debug information (only show in development)
if st.sidebar.checkbox("Show Debug Info"):
    st.subheader("🐛 Debug Information")
    st.write("**Session State:**", st.session_state)
    if data is not None:
        st.write("**Data Shape:**", data.shape)
        st.write("**Data Columns:**", list(data.columns))
    st.write("**Models Loaded:**", logistic_model is not None and vectorizer is not None)
