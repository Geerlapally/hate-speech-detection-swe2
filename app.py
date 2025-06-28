
import streamlit as st
import joblib
from utils.preprocess import clean_text
from utils.augment import obfuscate_text
from utils.bpe_encoder import tokenize_with_bpe, load_bpe_tokenizer
from utils.language import detect_and_translate
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# Load models and vectorizer
vectorizer = joblib.load("model/vectorizer.pkl")
classifier = joblib.load("model/classifier.pkl")
bpe_tokenizer = load_bpe_tokenizer("model/bpe_tokenizer.json")
bert_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

st.title("🛡️ Hate Speech Detection (SWE²++)")
st.markdown("Detects hate speech even in obfuscated or foreign-language text.")

text = st.text_area("Enter text to classify", height=150)
mode = st.radio("Select Model", ["⚡ Fast (TF-IDF + LR)", "🎯 Accurate (DistilBERT)"])
apply_translation = st.checkbox("🌐 Auto-detect and translate if not English", value=True)
apply_obfuscation = st.checkbox("🧪 Test with adversarial obfuscation", value=False)

if st.button("🔍 Detect"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        original_text = text.strip()
        if apply_translation:
            text = detect_and_translate(original_text)

        text = clean_text(text)
        if apply_obfuscation:
            text = obfuscate_text(text)

        if mode.startswith("⚡"):
            tokens = tokenize_with_bpe(bpe_tokenizer, [text])
            vect_text = vectorizer.transform(tokens)
            pred_proba = classifier.predict_proba(vect_text)[0]
            pred_label = classifier.classes_[pred_proba.argmax()]
            confidence = pred_proba.max()
        else:
            result = bert_classifier(original_text)[0]
            pred_label = result["label"]
            confidence = result["score"]

        st.markdown(f"**Prediction:** `{pred_label}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}`")

        if confidence < 0.7:
            st.info("⚠️ Low confidence — consider manual review.")

st.markdown("---")
st.markdown("Made with ❤️ for B.Tech Major Project")
