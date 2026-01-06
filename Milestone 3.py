

import pandas as pd
import re

# ---------- LOAD RAW DATASET ----------
file_path = r"C:\Users\reddy\OneDrive\Desktop\DT\data\emailclass.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print(df.head())
print("\nColumns:", df.columns)

# ---------- MANUAL STOPWORDS ----------
stop_words = {
    "the","is","at","which","on","and","a","an","in","to","for","of","this","that",
    "with","as","it","be","are","was","were","by","from","or","we","you","your",
    "about","but","not","have","has","had","they","their","them","he","she","his",
    "her","its","our","us"
}

# ---------- CLEANING FUNCTION ----------
def clean_email(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---------- APPLY CLEANING ----------
df["cleaned_text"] = df["text"].apply(clean_email)

# Fix NaN & ensure string type
df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)

# OPTIONAL: remove empty rows
df = df[df["cleaned_text"].str.strip() != ""]

print("\nCleaning completed! Preview:")
print(df[["text", "cleaned_text"]].head())

# ---------- SAVE CLEANED DATA ----------
output_path = r"C:\Users\reddy\OneDrive\Desktop\DT\data\cleaned_emailclass.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved at:\n{output_path}")


# MILESTONE 2: EMAIL CATEGORIZATION ENGINE 

import pandas as pd

# 1. LOAD CLEANED DATASET

file_path = r"C:\Users\reddy\OneDrive\Desktop\DT\data\cleaned_emailclass.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("Columns:", df.columns)

X = df["cleaned_text"]
y = df["type"]

# 2. TRAIN-TEST SPLIT + IMPROVED TF-IDF


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=15000,     # more expressive vocabulary
    ngram_range=(1, 3),     # unigrams + bigrams + trigrams
    min_df=2,               # remove very rare words
    max_df=0.9,             # remove very common words
    sublinear_tf=True       # log-scaled term frequency
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 3. BASELINE MODEL 1: LOGISTIC REGRESSION


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_tfidf, y_train)

lr_pred = lr.predict(X_test_tfidf)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))


# 4. BASELINE MODEL 2: NAIVE BAYES

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB(alpha=0.5)
nb.fit(X_train_tfidf, y_train)

nb_pred = nb.predict(X_test_tfidf)

print("\n--- Naive Bayes ---")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# 5. ADVANCED MODEL: TUNED LINEAR SVM


from sklearn.svm import LinearSVC

svm = LinearSVC(
    C=1.5,                  # tuned regularization
    class_weight="balanced",
    max_iter=5000
)

svm.fit(X_train_tfidf, y_train)

svm_pred = svm.predict(X_test_tfidf)

print("\n--- Linear SVM (Improved Advanced Model) ---")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# 6. SAMPLE PREDICTION

sample_email = ["Internet is not working, please resolve urgently"]
sample_vec = vectorizer.transform(sample_email)
prediction = svm.predict(sample_vec)

print("\nSample Email Prediction:", prediction[0])

# ======================================================
# IMPROVED URGENCY DETECTION & SCORING (ONE FRAME)
# ======================================================

from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ---------- WEIGHTED URGENCY KEYWORDS & PHRASES ----------
urgency_weights = {
    "urgent": 2,
    "urgently": 2,
    "immediately": 2,
    "asap": 2,
    "critical": 3,
    "emergency": 3,
    "priority": 2,
    "important": 1,
    "now": 1,
    "resolve": 1,
    "not working": 3,
    "immediate attention": 3,
    "needs attention": 2,
    "server down": 3,
    "system down": 3
}

# ---------- CREATE URGENCY LABEL (FOR TRAINING) ----------
def create_urgency_label(text):
    text = text.lower()
    for phrase in urgency_weights:
        if phrase in text:
            return 1
    return 0

df["urgency_label"] = df["cleaned_text"].apply(create_urgency_label)

print("\nUrgency Label Distribution:")
print(df["urgency_label"].value_counts())

# ---------- TRAIN-TEST SPLIT ----------
X_u = df["cleaned_text"]
y_u = df["urgency_label"]

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
    X_u,
    y_u,
    test_size=0.2,
    random_state=42,
    stratify=y_u
)

# ---------- TF-IDF (REUSE SAME SETTINGS) ----------
X_train_u_tfidf = vectorizer.fit_transform(X_train_u)
X_test_u_tfidf = vectorizer.transform(X_test_u)

# ---------- URGENCY ML MODEL ----------
urg_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

urg_model.fit(X_train_u_tfidf, y_train_u)
urg_pred = urg_model.predict(X_test_u_tfidf)

# ---------- EVALUATION ----------
print("\n--- Improved Urgency Detection ---")
print("F1 Score:", f1_score(y_test_u, urg_pred))
print("\nClassification Report:")
print(classification_report(y_test_u, urg_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_u, urg_pred))

# ---------- WEIGHTED KEYWORD SCORING ----------
def weighted_urgency_score(text):
    score = 0
    text = text.lower()
    for phrase, weight in urgency_weights.items():
        if phrase in text:
            score += weight
    return score

# ---------- FINAL URGENCY SCORE (ML + KEYWORDS) ----------
def final_urgency_score(text):
    text = text.lower()
    vec = vectorizer.transform([text])
    ml_score = urg_model.predict(vec)[0] * 3  # ML weight
    return ml_score + weighted_urgency_score(text)

# ---------- URGENCY LEVEL ----------
def urgency_level(score):
    if score >= 7:
        return "HIGH"
    elif score >= 4:
        return "MEDIUM"
    else:
        return "LOW"

# ---------- SAMPLE TEST ----------
samples = [
    "Internet is not working please resolve urgently",
    "Thank you for your support",
    "This is critical and needs immediate attention",
    "Server is down this is an emergency",
    "Please fix this asap"
]

print("\n--- Final Urgency Prediction ---")
for s in samples:
    score = final_urgency_score(s)
    print(f"\nEmail: {s}")
    print("Urgency Score:", score)
    print("Urgency Level:", urgency_level(score))
