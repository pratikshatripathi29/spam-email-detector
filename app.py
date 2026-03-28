from pathlib import Path
import pickle
import string

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


def clean_text(text):
    text = text.lower().strip()
    return "".join(char for char in text if char not in string.punctuation)


st.set_page_config(page_title="Spam Email Detector", page_icon="M")
st.title("Spam Email Detector")
st.write("Paste an email or message below to classify it as spam or not spam.")

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError as exc:
    st.error(f"Required file not found: {exc.filename}")
    st.stop()
except Exception as exc:
    st.error(f"Failed to load model files: {exc}")
    st.stop()

input_msg = st.text_area(
    "Enter your message",
    height=180,
    placeholder="Type or paste a message here...",
)

if st.button("Predict", type="primary"):
    cleaned = clean_text(input_msg)

    if not cleaned:
        st.warning("Please enter a message before running the prediction.")
        st.stop()

    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        st.error("Spam message detected.")
    else:
        st.success("This looks like a normal message.")

    if hasattr(model, "predict_proba"):
        spam_score = float(model.predict_proba(vectorized)[0][1])
        st.caption(f"Spam confidence: {spam_score:.1%}")
