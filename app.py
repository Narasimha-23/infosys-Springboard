# =============================
# STREAMLIT SMART EMAIL URGENCY
# =============================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time

# -----------------------------
# LOAD MODELS
# -----------------------------
vectorizer = joblib.load("tfidf_vectorizer.pkl")
category_model = joblib.load("svm_email_classifier.pkl")
urgency_model = joblib.load("urgency_model.pkl")

# -----------------------------
# URGENCY FUNCTIONS
# -----------------------------
urgency_keywords = {
    "urgent": 2,
    "urgently": 2,
    "critical": 3,
    "emergency": 3,
    "asap": 2,
    "server down": 3,
    "not working": 3,
    "immediately": 2,
    "failure": 3
}

def urgency_score(text):
    score = 0
    text_lower = text.lower()
    # Keyword-based scoring
    for word, weight in urgency_keywords.items():
        if word in text_lower:
            score += weight
    # ML-based scoring
    vec = vectorizer.transform([text])
    score += urgency_model.predict(vec)[0] * 3
    return score

def urgency_level(score):
    if score >= 7:
        return "HIGH"
    elif score >= 4:
        return "MEDIUM"
    else:
        return "LOW"

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📧 Smart Email Urgency Detection")

# ----- Single Email Input -----
st.subheader("✏️ Check Urgency of Your Email")
email_input = st.text_area(
    "Enter email text here",
    placeholder="Example: Internet is not working, please resolve urgently"
)

if st.button("Analyze Email"):
    if email_input.strip() == "":
        st.warning("Please enter an email text!")
    else:
        score = urgency_score(email_input)
        level = urgency_level(score)

        st.write("### 🔍 Results")
        st.info(f"📊 Urgency Score: {score}")
        if level == "HIGH":
            st.error(f"🚦 Urgency Level: {level}")
        elif level == "MEDIUM":
            st.warning(f"🚦 Urgency Level: {level}")
        else:
            st.success(f"🚦 Urgency Level: {level}")

# ----- Live Email Stream -----
st.subheader("🔁 Simulated Live Email Stream")
live_emails = [
    "Server is down, this is critical",
    "Internet is not working please resolve",
    "Need access to VPN",
    "Thank you for the update",
    "Urgent: system failure detected"
]

if st.button("Start Live Stream"):
    for mail in live_emails:
        score = urgency_score(mail)
        level = urgency_level(score)

        st.write("📧 Email:", mail)
        st.write("⚠️ Urgency Score:", score)
        if level == "HIGH":
            st.error(f"🚦 Urgency Level: {level}")
        elif level == "MEDIUM":
            st.warning(f"🚦 Urgency Level: {level}")
        else:
            st.success(f"🚦 Urgency Level: {level}")
        
        st.divider()
        time.sleep(2)
