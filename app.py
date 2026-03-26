import streamlit as st
import pickle
import string

# Load model
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

# UI
st.title("📧 Spam Email Detector")

input_msg = st.text_area("Enter your message")

if st.button("Predict"):
    cleaned = clean_text(input_msg)
    vectorized = cv.transform([cleaned])
    result = model.predict(vectorized)[0]

    if result == 1:
        st.error("🚫 Spam Message")
    else:
        st.success("✅ Not Spam (Ham)")