AI Powered Smart Email Classifier for Enterprises
Project Overview:

This project implements an AI-powered system for enterprises to automatically clean, classify, and prioritize emails.
The system reduces manual effort by categorizing emails and identifying urgent communications using machine learning and intelligent rule-based techniques.

Milestone 1: Email Data Cleaning & Preprocessing

Objective: Prepare raw enterprise email data for machine learning.

Key Tasks:

Remove HTML tags, URLs, email IDs, numbers, and special characters

Convert text to lowercase

Remove stopwords

Handle missing and empty values

Milestone 2: Email Classification Engine

Objective: Automatically classify emails into predefined categories.

Approach:

TF-IDF text vectorization

Machine learning models:

Logistic Regression

Naive Bayes

Linear SVM (final selected model)

Evaluation Metrics:
Accuracy and Classification Report

Milestone 3: Urgency Detection & Scoring

Objective: Identify and prioritize urgent enterprise emails.

Approach:

Logistic Regression-based urgency classifier

Hybrid urgency detection using:

Weighted urgency keywords and phrases

Capitalization and punctuation emphasis

Negative sentiment indicators

Email length features
