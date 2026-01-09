from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
import re
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler
from ml.preprocess import clean_text

app = FastAPI(title="Stack Overflow Engagement Predictor")

ARTIFACT_PATH = Path("artifacts/pipeline.joblib")
bundle = joblib.load(ARTIFACT_PATH)

model = bundle["model"]
text_vectorizer = bundle["text_vectorizer"]
tag_vectorizer = bundle["tag_vectorizer"]
scaler = bundle["scaler"]
threshold = float(bundle["threshold"])
meta_cols = bundle["meta_cols"]

def tags_to_string(tags):
    # tags arrive as ["python","pandas"] -> "python pandas"
    return " ".join([t.strip().lower() for t in tags if t and t.strip()])

def build_meta(req: PredictRequest) -> np.ndarray:
    title = req.title or ""
    body = req.body or ""

    title_len = min(len(title), 200)
    body_len = min(len(body), 2000)
    num_tags = len(req.tags) if req.tags else 0
    has_code_block = 1 if "<code>" in (body or "").lower() else 0

    hour = int(req.hour) if req.hour is not None else 12
    weekday = int(req.weekday) if req.weekday is not None else 2

    # must match meta_cols order from training
    meta = np.array([[hour, weekday, title_len, body_len, num_tags, has_code_block]], dtype=float)
    return meta

def is_valid_question(title: str, body: str, tags: list[str]) -> tuple[bool, list[str]]:
    reasons = []
    title = (title or "").strip()
    body = (body or "").strip()
    tags = tags or []

    if len(title) < 8:
        reasons.append("title_too_short")
    if len(body) < 30:
        reasons.append("body_too_short")
    if len(tags) == 0:
        reasons.append("no_tags")

    # require some alphabetic signal
    alpha = sum(c.isalpha() for c in (title + " " + body))
    if alpha < 20:
        reasons.append("too_few_alphabetic_chars")

    # require at least a few word tokens
    tokens = re.findall(r"[a-zA-Z]{2,}", title + " " + body)
    if len(tokens) < 10:
        reasons.append("too_few_words")

    # low diversity = likely garbage
    if len(tokens) > 0:
        diversity = len(set(tokens)) / len(tokens)
        if diversity < 0.15:
            reasons.append("low_word_diversity")

    return (len(reasons) == 0), reasons

def top_feature_contributions(X_text_tfidf, X_tags_tfidf, X_meta_scaled, k=6):
    """
    For a single sample, compute top contributing features.
    We’ll report a mix of text + tags + meta.
    """
    coefs = model.coef_[0]

    text_names = text_vectorizer.get_feature_names_out()
    tag_names = tag_vectorizer.get_feature_names_out()

    text_dim = len(text_names)
    tag_dim = len(tag_names)
    meta_dim = X_meta_scaled.shape[1]

    # contributions = value * weight
    # for sparse matrices, use elementwise multiply then get top nonzeros
    text_contrib = X_text_tfidf.multiply(coefs[:text_dim]).toarray().ravel()
    tag_contrib = X_tags_tfidf.multiply(coefs[text_dim:text_dim + tag_dim]).toarray().ravel()
    meta_contrib = (X_meta_scaled.toarray().ravel() * coefs[text_dim + tag_dim:text_dim + tag_dim + meta_dim])

    drivers = []

    # top text
    if text_contrib.size:
        idx = np.argsort(text_contrib)[-k:]
        for i in reversed(idx):
            if text_contrib[i] > 0:
                drivers.append(f"text:{text_names[i]}")

    # top tags
    if tag_contrib.size:
        idx = np.argsort(tag_contrib)[-k:]
        for i in reversed(idx):
            if tag_contrib[i] > 0:
                drivers.append(f"tag:{tag_names[i]}")

    # meta drivers (by abs)
    meta_names = meta_cols
    # pick by absolute contribution, but skip near-zero
    idx = np.argsort(np.abs(meta_contrib))[-k:]
    for i in reversed(idx):
        if abs(meta_contrib[i]) > 1e-6:
            drivers.append(f"meta:{meta_names[i]}")


    # de-dup and truncate
    seen = set()
    out = []
    for d in drivers:
        if d not in seen:
            out.append(d)
            seen.add(d)
        if len(out) >= 10:
            break
    return out

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    print("HAS_CODE:", "<code>" in (req.body or "").lower())
    ok, reasons = is_valid_question(req.title, req.body, req.tags)
    if not ok:
        return {
        "will_get_answered": False,
        "probability_answered": 0.0,
        "threshold": threshold,
        "top_drivers": reasons
        }

    text = clean_text((req.title or "") + " " + (req.body or ""))
    tags_str = tags_to_string(req.tags)

    X_text_tfidf = text_vectorizer.transform([text])
    X_tags_tfidf = tag_vectorizer.transform([tags_str])

    meta = build_meta(req)
    meta_scaled = scaler.transform(meta)
    X_meta_sparse = csr_matrix(meta_scaled)

    X = hstack([X_text_tfidf, X_tags_tfidf, X_meta_sparse])

    prob_answered = float(model.predict_proba(X)[0, 1])
    will_get_answered = prob_answered >= threshold

    drivers = top_feature_contributions(X_text_tfidf, X_tags_tfidf, X_meta_sparse, k=5)

    return PredictResponse(
        will_get_answered=bool(will_get_answered),
        probability_answered=prob_answered,
        threshold=threshold,
        top_drivers=drivers
    )
