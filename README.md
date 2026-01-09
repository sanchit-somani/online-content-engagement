
# Stack Overflow Engagement Predictor (Posting-time)

Predict whether a Stack Overflow question will receive at least one answer **using only posting-time information**:
- Title + Body text (TF-IDF)
- Tags (TF-IDF on tag tokens)
- Posting time features (hour, weekday)
- Simple clarity proxies (title/body length, number of tags, code-block presence)

This repo includes:
- Training pipeline (`ml/train.py`) that saves a reproducible artifact bundle
- FastAPI service (`app/main.py`) that loads artifacts and serves predictions

---

## Quickstart (60 seconds)

```bash
make venv
make install
make train
make serve
```

Then open:
**http://127.0.0.1:8000/docs**

Try the /predict endpoint with:

{
  "title": "Why does my SQL query return duplicate rows?",
  "body": "I have two tables and a join... <code>SELECT ...</code>",
  "tags": ["sql", "mysql"],
  "hour": 14,
  "weekday": 2
}

# Model framing
## Target

answered = (answer_count > 0)
Binary classification: will the question receive at least one answer?

## Why threshold = 0.6?

I treat thresholding as a product decision. A higher threshold increases how often the system flags questions as likely unanswered, which improves recall for the unanswered class at the cost of more false positives.


