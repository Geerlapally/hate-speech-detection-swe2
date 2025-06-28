import streamlit as st
import pandas as pd
from utils.language import detect_and_translate
from utils.model import load_model, predict_label

# Title and Description
st.set_page_config(page_title="Hate Speech Detection - SWE²++", layout="wide")
st.title("🚨 Hate Speech Detection using SWE²++")
st.markdown(
    "This tool detects hate speech in social media posts using both traditional ML and Transformer models. "
    "It handles obfuscation and multilingual text using subword modeling and translation."
)

# Load dataset from Google Drive (direct download link)
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?export=download&id=14AeGLMqjh-cYjz6a_XoXV91EWPFhGDcu"
    df = pd.read_csv(url)
    return df

data = load_data()

# Show data sample
if st.checkbox("📊 Show Sample Dataset"):
    st.write(data.sample(5))

# Load models
logistic_model, vectorizer = load_model()

# Text input
user_input = st.text_area("✍️ Enter a tweet or message to analyze:")

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Detect language and translate if needed
        translated_text = detect_and_translate(user_input)
        st.write("🌐 Detected/Translated Text:", translated_text)

        # Predict label
        label = predict_label(translated_text, logistic_model, vectorizer)

        # Show result
        label_map = {
            0: "Hate Speech ❌",
            1: "Offensive 🛑",
            2: "Neutral ✅"
        }
        st.markdown(f"### 📢 Prediction: **{label_map.get(label, 'Unknown')}**")
