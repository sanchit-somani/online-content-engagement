import re

STOPWORDS_SMALL = {
    "the","and","or","to","of","in","for","on","with","from","by","as","at",
    "is","are","was","were","be","been","being","it","this","that","these","those",
    "i","you","we","they","my","your","our","their",
    "what","why","how","when","where",
    "a","an"
}

def quality_features(title: str, body: str) -> dict:
    text = (title or "") + " " + (body or "")
    text_l = text.lower()

    tokens = re.findall(r"[a-zA-Z]{2,}", text_l)
    n = len(tokens)

    letters = re.findall(r"[a-zA-Z]", text_l)
    n_letters = len(letters)
    vowels = sum(c in "aeiou" for c in letters)
    vowel_ratio = (vowels / n_letters) if n_letters else 0.0

    stop_hits = sum(t in STOPWORDS_SMALL for t in tokens)
    stopword_rate = (stop_hits / n) if n else 0.0

    long_token_rate = (sum(len(t) >= 12 for t in tokens) / n) if n else 0.0

    return {
        "stopword_rate": stopword_rate,
        "vowel_ratio": vowel_ratio,
        "long_token_rate": long_token_rate,
    }

def quality_score(q: dict) -> float:
    """
    1.0 = looks like normal language
    0.0 = very likely nonsense / low-quality
    Simple heuristic: penalize low stopwords, low vowel ratio, high long-token rate.
    """
    score = 1.0
    if q["stopword_rate"] < 0.02:
        score -= 0.4
    if q["vowel_ratio"] < 0.25:
        score -= 0.3
    if q["long_token_rate"] > 0.5:
        score -= 0.2
    return max(0.0, min(1.0, score))
