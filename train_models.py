import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# LOAD DATA
df = pd.read_csv(r"C:\Users\reddy\OneDrive\Desktop\DT\data\cleaned_emailclass.csv")

X = df["cleaned_text"]
y = df["type"]

# VECTORIZE
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,3))
X_tfidf = vectorizer.fit_transform(X)

# CATEGORY MODEL
svm = LinearSVC()
svm.fit(X_tfidf, y)

# URGENCY LABEL
df["urgency_label"] = df["cleaned_text"].apply(
    lambda x: 1 if any(w in x.lower() for w in ["urgent","critical","asap","down"]) else 0
)

# URGENCY MODEL
urg_model = LogisticRegression(max_iter=2000)
urg_model.fit(X_tfidf, df["urgency_label"])

# SAVE ALL FILES IN SAME FOLDER
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(svm, "svm_email_classifier.pkl")
joblib.dump(urg_model, "urgency_model.pkl")

print("✅ All models saved successfully!")
