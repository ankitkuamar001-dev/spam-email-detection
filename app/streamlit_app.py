import streamlit as st
import pickle
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run training script first.")
    st.stop()

# --- Preprocessing Function (Must match training!) ---
# Duplicating here to avoid import issues if deployed separately, 
# or could import from src.preprocessing if structure allows.
# For a standalone app, keeping it self-contained is often safer.

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stemmer = PorterStemmer()
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
    clean_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

# --- App Layout ---
st.set_page_config(page_title="Spam Detector", page_icon="🚫")

st.title("📩 SMS Spam Detection")
st.markdown("Enter a message below to check if it's **Spam** or **Ham** (Not Spam).")

input_text = st.text_area("Message Content", height=150, placeholder="Type your message here...")

if st.button("Analyze Message"):
    if input_text:
        # Preprocess
        cleaned_text = clean_text(input_text)
        
        # Vectorize
        features = vectorizer.transform([cleaned_text]).toarray()
        
        # Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Display Result
        if prediction == 'spam':
            st.error(f"🚨 **SPAM DETECTED** (Confidence: {probabilities[1]*100:.2f}%)")
        else:
            st.success(f"✅ **HAM (Safe)** (Confidence: {probabilities[0]*100:.2f}%)")
            
        with st.expander("See details"):
            st.write(f"**Cleaned Text:** {cleaned_text}")
            st.write(f"**Probabilities:** Ham: {probabilities[0]:.4f}, Spam: {probabilities[1]:.4f}")
            
    else:
        st.warning("Please enter some text to analyze.")

st.sidebar.info("Model: Naive Bayes / Random Forest (Best Performer)")
