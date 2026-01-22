#  AI-Powered Smart Email Classifier for Enterprises

## Project Overview
This project implements a **Smart Email Urgency Detection System** that classifies incoming emails by category and predicts their urgency level. The system combines **Machine Learning**, **keyword-based scoring**, and a **Streamlit interactive dashboard** to provide enterprise-ready email management.

---

## 🔹 Features

- **Email Categorization:** Classifies emails into categories using SVM + TF-IDF.
- **Urgency Detection:** Predicts urgency levels (LOW, MEDIUM, HIGH) using ML + keyword scoring.
- **Interactive Dashboard:** Streamlit UI to input emails and check urgency in real-time.
- **Live Email Stream:** Simulates incoming emails with dynamic urgency detection.
- **Deployment Ready:** Can be deployed on **Streamlit Cloud / Azure / AWS / GCP**.

---

##  Milestones & Timeline (8 Weeks)

### **Milestone 1 (Weeks 1–2): Data Collection & Preprocessing**
- **Objective:** Prepare labeled dataset for training.
- **Tasks Completed:**
  - Collected and cleaned email datasets.
  - Removed noise (HTML tags, signatures, numbers, special characters).
  - Labeled emails with categories (`type`) and urgency (`urgency_label`).

### **Milestone 2 (Weeks 3–4): Email Categorization Engine**
- **Objective:** Develop an NLP-based classification system.
- **Tasks Completed:**
  - Trained baseline classifiers: Logistic Regression, Naive Bayes.
  - Implemented **SVM with TF-IDF** for improved classification.
  - Evaluated model accuracy and classification reports.

### **Milestone 3 (Weeks 5–6): Urgency Detection & Scoring**
- **Objective:** Implement urgency prediction.
- **Tasks Completed:**
  - Trained urgency classification model (Logistic Regression).
  - Combined ML predictions with **keyword-based scoring**.
  - Validated results using **F1 score** and **confusion matrix**.

### **Milestone 4 (Weeks 7–8): Dashboard & Deployment**
- **Objective:** Deliver enterprise-ready solution.
- **Tasks Completed:**
  - Built an interactive Streamlit dashboard with email input box.
  - Implemented **simulated live email stream**.
  - Displayed urgency scores and levels dynamically.
  - Deployment-ready for **Streamlit Cloud** (or cloud platforms like Azure/AWS/GCP)

