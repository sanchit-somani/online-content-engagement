import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from preprocess import clean_text


df = pd.read_csv("data/questions.csv")

# print(df.head(20))
# df.info()

# class balance check 
df["answered"] = (df["answer_count"] > 0).astype(int)
# print(df["answered"].value_counts(normalize=True))

# text processing

df["text"] = df["title"] + " " + df["body"]
df["clean_text"] = df["text"].apply(clean_text)

df["creation_date"] = pd.to_datetime(df["creation_date"], errors="coerce")
df["hour"] = df["creation_date"].dt.hour.fillna(0).astype(int)
df["weekday"] = df["creation_date"].dt.weekday.fillna(0).astype(int)

def parse_tags(tag_str: str) -> str:
    if pd.isna(tag_str):
        return ""
    s = str(tag_str).strip()
    # common format: "<python><pandas>"
    if "<" in s and ">" in s:
        s = s.replace("><", " ").replace("<", "").replace(">", "")
        return s
    # fallback: already space/comma separated
    return s.replace(",", " ")

df["tags_text"] = df["tags"].apply(parse_tags)

meta_cols = ["comment_count", "favorite_count", "hour", "weekday", "score", "view_count"]
X_meta = df[meta_cols].copy()

X_text = df["clean_text"]
X_tags = df["tags_text"]
X_time = df[["hour", "weekday"]]
y = df["answered"]

# train-test split 

X_text_train, X_text_test, X_tags_train, X_tags_test, X_time_train, X_time_test, y_train, y_test = train_test_split(
    X_text, X_tags, X_time, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# X_train, X_test, y_train, y_test = train_test_split(
#     df["clean_text"],
#     df["answered"],
#     test_size=0.2,
#     random_state=42,
#     stratify=df["answered"]
# )

# X_train = X_train.reset_index(drop=True)
# X_test  = X_test.reset_index(drop=True)
# y_train = y_train.reset_index(drop=True)
# y_test  = y_test.reset_index(drop=True)

# TF-IDF on text

text_vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2
)
X_text_train_tfidf = text_vectorizer.fit_transform(X_text_train)
X_text_test_tfidf = text_vectorizer.transform(X_text_test)

# TF-IDF on tags

tag_vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 1),
    lowercase=True
)
X_tags_train_tfidf = tag_vectorizer.fit_transform(X_tags_train)
X_tags_test_tfidf = tag_vectorizer.transform(X_tags_test)

# scale metadata (important for logistic regression)
scaler = StandardScaler()
X_time_train_scaled = scaler.fit_transform(X_time_train)
X_time_test_scaled = scaler.transform(X_time_test)

X_time_train_sparse = csr_matrix(X_time_train_scaled)
X_time_test_sparse = csr_matrix(X_time_test_scaled)

# combining features

X_train_final = hstack([X_text_train_tfidf, X_tags_train_tfidf, X_time_train_sparse])
X_test_final = hstack([X_text_test_tfidf, X_tags_test_tfidf, X_time_test_sparse])

#logistic regression

model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train_final, y_train)

# precision, recall, f1

y_probs = model.predict_proba(X_test_final)[:, 1]
y_pred = (y_probs >= 0.5).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# insights

coefs = model.coef_[0]

text_names = text_vectorizer.get_feature_names_out()
tag_names = tag_vectorizer.get_feature_names_out()

text_dim = len(text_names)
tag_dim = len(tag_names)
time_dim = 2  # hour, weekday

# slice coef blocks
text_coefs = coefs[:text_dim]
tag_coefs = coefs[text_dim:text_dim + tag_dim]
time_coefs = coefs[text_dim + tag_dim:text_dim + tag_dim + time_dim]

# top words (answered vs unanswered) from main text only
top_pos = np.argsort(text_coefs)[-20:]
top_neg = np.argsort(text_coefs)[:20]

print("\nTop TEXT features pushing toward answered:")
print(text_names[top_pos])

print("\nTop TEXT features pushing toward unanswered:")
print(text_names[top_neg])

# top tags
top_tag_pos = np.argsort(tag_coefs)[-15:]
top_tag_neg = np.argsort(tag_coefs)[:15]

print("\nTop TAGS pushing toward answered:")
print(tag_names[top_tag_pos])

print("\nTop TAGS pushing toward unanswered:")
print(tag_names[top_tag_neg])

print("\nTime feature weights (hour, weekday):")
print(time_coefs)
