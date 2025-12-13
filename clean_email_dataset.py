<<<<<<< HEAD
import pandas as pd
import re

# ---------- LOAD DATASET ----------
file_path = r"C:\Users\reddy\OneDrive\Desktop\DT\data\emailclass.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print(df.head())


# ---------- MANUAL STOPWORDS LIST ----------
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

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove Email IDs
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation and special chars
    text = re.sub(r"[^\w\s]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = [w for w in text.split() if w not in stop_words]

    return " ".join(words)


# ---------- APPLY CLEANING ----------
df["cleaned_text"] = df["text"].apply(clean_email)

print("\nCleaning completed! Preview:")
print(df.head())


# ---------- EXPORT CLEANED FILE ----------
output_path = r"C:\Users\reddy\OneDrive\Desktop\DT\data\cleaned_emailclass.csv"
df.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved successfully at:\n{output_path}")
=======
import pandas as pd
import re

# ---------- LOAD DATASET ----------
file_path = r"C:\Users\reddy\OneDrive\Desktop\DT\data\emailclass.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print(df.head())


# ---------- MANUAL STOPWORDS LIST ----------
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

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove Email IDs
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation and special chars
    text = re.sub(r"[^\w\s]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = [w for w in text.split() if w not in stop_words]

    return " ".join(words)


# ---------- APPLY CLEANING ----------
df["cleaned_text"] = df["text"].apply(clean_email)

print("\nCleaning completed! Preview:")
print(df.head())


# ---------- EXPORT CLEANED FILE ----------
output_path = r"C:\Users\reddy\OneDrive\Desktop\DT\data\cleaned_emailclass.csv"
df.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved successfully at:\n{output_path}")
>>>>>>> b33cc985724a0f933f911c27732bca47037c76c3
