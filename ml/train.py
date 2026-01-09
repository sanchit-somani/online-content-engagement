"""
Train a posting-time engagement model for Stack Overflow questions.

Goal:
Predict whether a question will receive at least one answer using only
information available at posting time.

Features:
- Text (title + body) via TF-IDF
- Tags via TF-IDF
- Numeric posting-time features:
    * time (hour, weekday)
    * clarity proxies (lengths, code presence)
    * quality heuristics (stopword rate, vowel ratio, long-token rate)

Artifacts saved:
- model
- text vectorizer
- tag vectorizer
- scaler for numeric features
- classification threshold
- ordered list of numeric meta columns
"""

# =========================
# Imports
# =========================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from scipy.sparse import hstack, csr_matrix

from ml.preprocess import clean_text
from ml.quality import quality_features


# =========================
# Load data
# =========================

df = pd.read_csv("data/questions.csv")

# Binary target: answered or not
df["answered"] = (df["answer_count"] > 0).astype(int)


# =========================
# Text preprocessing
# =========================

# Safe concatenation (handles NaNs)
df["text"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["body"].fillna("").astype(str)
)

df["clean_text"] = df["text"].apply(clean_text)


# =========================
# Time features (posting-time)
# =========================

df["creation_date"] = pd.to_datetime(df["creation_date"], errors="coerce")
df["hour"] = df["creation_date"].dt.hour.fillna(0).astype(int)
df["weekday"] = df["creation_date"].dt.weekday.fillna(0).astype(int)


# =========================
# Tag preprocessing
# =========================

def parse_tags(tag_str: str) -> str:
    """
    Convert Stack Overflow tag formats like:
        <python><pandas>
    into:
        python pandas
    """
    if pd.isna(tag_str):
        return ""
    s = str(tag_str).strip()
    if "<" in s and ">" in s:
        return s.replace("><", " ").replace("<", "").replace(">", "")
    return s.replace(",", " ")

df["tags_text"] = df["tags"].apply(parse_tags)


# =========================
# Clarity / effort features (posting-time)
# =========================

# Lengths are capped to avoid spammy inflation
df["title_len"] = df["title"].fillna("").astype(str).str.len().clip(upper=200)
df["body_len"]  = df["body"].fillna("").astype(str).str.len().clip(upper=2000)

df["num_tags"] = df["tags_text"].apply(
    lambda s: 0 if not s else len(s.split())
)

df["has_code_block"] = (
    df["body"]
    .fillna("")
    .astype(str)
    .str.contains("<code>", regex=False)
    .astype(int)
)


# =========================
# Quality heuristics (posting-time)
# =========================

# stopword_rate, vowel_ratio, long_token_rate
quality = df.apply(
    lambda r: quality_features(
        r.get("title", ""),
        r.get("body", "")
    ),
    axis=1
)

df = pd.concat([df, pd.DataFrame(list(quality))], axis=1)


# =========================
# Define model inputs
# =========================

X_text = df["clean_text"]
X_tags = df["tags_text"]

meta_num_cols = [
    "hour",
    "weekday",
    "title_len",
    "body_len",
    "num_tags",
    "has_code_block",
    "stopword_rate",
    "vowel_ratio",
    "long_token_rate",
]

X_meta = df[meta_num_cols].copy()
y = df["answered"]


# =========================
# Train / test split
# =========================

(
    X_text_train,
    X_text_test,
    X_tags_train,
    X_tags_test,
    X_meta_train,
    X_meta_test,
    y_train,
    y_test,
) = train_test_split(
    X_text,
    X_tags,
    X_meta,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# =========================
# Vectorization
# =========================

# Text TF-IDF
text_vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
)

X_text_train_tfidf = text_vectorizer.fit_transform(X_text_train)
X_text_test_tfidf = text_vectorizer.transform(X_text_test)


# Tag TF-IDF (treated as tokens)
tag_vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 1),
    lowercase=True,
    use_idf=False,
    norm="l2",
)

X_tags_train_tfidf = tag_vectorizer.fit_transform(X_tags_train)
X_tags_test_tfidf = tag_vectorizer.transform(X_tags_test)


# =========================
# Scale numeric meta features
# =========================

scaler = StandardScaler()
X_meta_train_scaled = scaler.fit_transform(X_meta_train)
X_meta_test_scaled = scaler.transform(X_meta_test)

X_meta_train_sparse = csr_matrix(X_meta_train_scaled)
X_meta_test_sparse = csr_matrix(X_meta_test_scaled)


# =========================
# Combine all features
# =========================

X_train_final = hstack([
    X_text_train_tfidf,
    X_tags_train_tfidf,
    X_meta_train_sparse,
])

X_test_final = hstack([
    X_text_test_tfidf,
    X_tags_test_tfidf,
    X_meta_test_sparse,
])


# =========================
# Train model
# =========================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
)

model.fit(X_train_final, y_train)


# =========================
# Evaluate (threshold = 0.5 for reporting)
# =========================

y_probs = model.predict_proba(X_test_final)[:, 1]
y_pred = (y_probs >= 0.5).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# =========================
# Interpretability
# =========================

coefs = model.coef_[0]

text_names = text_vectorizer.get_feature_names_out()
tag_names = tag_vectorizer.get_feature_names_out()

text_dim = len(text_names)
tag_dim = len(tag_names)
meta_dim = len(meta_num_cols)

text_coefs = coefs[:text_dim]
tag_coefs = coefs[text_dim : text_dim + tag_dim]
meta_coefs = coefs[text_dim + tag_dim : text_dim + tag_dim + meta_dim]

print("\nTop TEXT features pushing toward answered:")
print(text_names[np.argsort(text_coefs)[-20:]])

print("\nTop TEXT features pushing toward unanswered:")
print(text_names[np.argsort(text_coefs)[:20]])

print("\nTop TAGS pushing toward answered:")
print(tag_names[np.argsort(tag_coefs)[-15:]])

print("\nTop TAGS pushing toward unanswered:")
print(tag_names[np.argsort(tag_coefs)[:15]])

print("\nMeta feature weights:")
for name, w in zip(meta_num_cols, meta_coefs):
    print(f"{name:18s} {w: .4f}")


# =========================
# Save artifacts
# =========================

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

bundle = {
    "model": model,
    "text_vectorizer": text_vectorizer,
    "tag_vectorizer": tag_vectorizer,
    "scaler": scaler,
    "threshold": 0.6,
    "meta_cols": meta_num_cols,
}

joblib.dump(bundle, ARTIFACT_DIR / "pipeline.joblib")
print("Saved artifacts to artifacts/pipeline.joblib")
