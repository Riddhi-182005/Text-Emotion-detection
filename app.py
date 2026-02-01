import streamlit as st
from transformers import pipeline

st.title("Text Emotion Detector")

st.write("Type a sentence and the model will detect the emotion.")

# Load pre-trained emotion model
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

text = st.text_input("Enter your text here")

if text:
    predictions = emotion_model(text)[0]

    best_emotion = max(predictions, key=lambda x: x["score"])

    st.subheader("Result")
    st.write(f"Emotion: {best_emotion['label']}")
    st.write(f"Confidence: {best_emotion['score']*100:.2f}%")
