import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
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

# train-test split 

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["answered"],
    test_size=0.2,
    random_state=42,
    stratify=df["answered"]
)

# TF-IDF 

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#logistic regression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# precision, recall, f1

y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# insights

feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]

top_positive = np.argsort(coefs)[-20:]
top_negative = np.argsort(coefs)[:20]

print("Words linked to answered questions:")
print(feature_names[top_positive])

print("\nWords linked to unanswered questions:")
print(feature_names[top_negative])




