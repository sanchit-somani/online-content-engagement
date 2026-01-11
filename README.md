
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




# Stack Overflow Engagement Predictor (Posting-time)

Predict whether a Stack Overflow question will receive at least one answer **using only information available at posting time**.

This project focuses on modeling *question quality and timing*, without using any post-hoc engagement signals (no leakage).

---

## What this model uses

**Text**
- Title + body (TF-IDF, unigrams + bigrams)

**Tags**
- Tag tokens vectorized separately (TF-IDF-style)

**Posting-time metadata**
- Hour of day
- Day of week

**Clarity / effort proxies**
- Title length (capped)
- Body length (capped)
- Number of tags
- Code block presence

**Quality guardrail features**
- Stopword rate
- Vowel ratio
- Long-token rate

---

## Output

The API returns both **raw** and **quality-adjusted** probabilities:

- `probability_answered` – raw model probability
- `quality_score` – heuristic score ∈ [0,1] measuring input quality
- `adjusted_probability_answered = probability_answered × quality_score`
- `will_get_answered` – decision using adjusted probability
- `valid_input` + `validation_reasons` – soft validation flags

This avoids overconfident predictions on empty or nonsensical inputs.

---

## Quickstart (60 seconds)

```bash
make venv
make install
make train
make serve
```

Then open:
```
http://127.0.0.1:8000/docs
```

---

## Example: single prediction

### Request
```json
{
  "title": "Why does my SQL query return duplicate rows?",
  "body": "I have two tables and a join... <code>SELECT ...</code>",
  "tags": ["sql", "mysql"],
  "hour": 14,
  "weekday": 2
}
```

### Response (abridged)
```json
{
  "valid_input": true,
  "will_get_answered": true,
  "probability_answered": 0.74,
  "quality_score": 1.0,
  "adjusted_probability_answered": 0.74,
  "threshold": 0.6,
  "top_drivers": ["text:sql", "tag:mysql", "meta:body_len"]
}
```

---

## Batch prediction

You can score multiple questions at once using `/predict_batch`.

### Request
```json
{
  "items": [
    {
      "title": "SQL join duplicates rows",
      "body": "I have two tables ... <code>SELECT ...</code>",
      "tags": ["sql"],
      "hour": 14,
      "weekday": 2
    },
    {
      "title": "",
      "body": "",
      "tags": [],
      "hour": 12,
      "weekday": 2
    }
  ]
}
```

---

## Model framing

### Target
```text
answered = (answer_count > 0)
```

Binary classification: will the question receive at least one answer?

### Threshold choice
The default threshold is **0.6**, treated as a product decision:
- Higher threshold → better recall for unanswered questions
- Lower threshold → more optimistic predictions

The final decision uses **adjusted probability**, not raw probability.

---

## Results (5k sample, 80/20 split)

Test set size: 1000  
Class distribution: 132 unanswered / 868 answered

### Confusion matrices (raw probability)

**Threshold = 0.5**
```
[[ 53  79]
 [ 95 773]]
```

**Threshold = 0.6**
```
[[ 69  63]
 [224 644]]
```

**Threshold = 0.7**
```
[[ 90  42]
 [402 466]]
```

---

## Demo script

Run an end-to-end demo (train → serve → sample requests):

```bash
bash scripts/demo.sh
```

---

## Repo structure

```
ml/
  train.py           # training + artifact generation
  preprocess.py      # text cleaning
  quality.py         # quality features + scoring

app/
  main.py            # FastAPI service
  schemas.py         # request / response models

artifacts/
  pipeline.joblib    # trained model + vectorizers + scaler

scripts/
  demo.sh            # end-to-end demo
```

