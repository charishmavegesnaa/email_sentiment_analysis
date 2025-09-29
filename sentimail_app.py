import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page title
st.title("SentiMail Sentiment Prediction App")

# Load the trained logistic regression model and vectorizer
try:
    with open('sentimail_logreg_model.pkl', 'rb') as file:
        data = pickle.load(file)
        logreg = data['model']
        tfidf = data['vectorizer']
except FileNotFoundError:
    st.error("Model file 'sentimail_logreg_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Centered input and button (always visible after model loads)
st.write("""
<div style='display: flex; flex-direction: column; align-items: center;'>
    <h3>Enter Email Topic</h3>
</div>
""", unsafe_allow_html=True)

email_topic = st.text_area("Email Topic", key="email_topic", help=None)

predict_btn = st.button("Sentiment Analysis")

if predict_btn:
    if not email_topic.strip():
        st.error("Please enter an email topic.")
    else:
        # Transform input using the loaded TF-IDF vectorizer
        X_input = tfidf.transform([email_topic])
        # Make prediction
        prediction = logreg.predict(X_input)[0]
        # Map prediction to label
        label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        predicted_label = label_map.get(prediction, "Unknown")
        # Display result
        st.subheader("Prediction Result")
        st.markdown(f"<div style='text-align:center;font-size:24px;'><b>{predicted_label}</b></div>", unsafe_allow_html=True)

# Display instructions
st.write("""
### Instructions
1. Enter the email topic or content above.
2. Click the 'Sentiment Analysis' button to see the predicted sentiment category.
""")

