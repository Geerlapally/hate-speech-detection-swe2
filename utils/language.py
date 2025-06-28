
from langdetect import detect
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != 'en':
            result = translator(text)
            return result[0]['translation_text']
        return text
    except:
        return text
