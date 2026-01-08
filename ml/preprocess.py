import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)   # remove HTML
    text = re.sub(r"http\S+", " ", text) # remove URLs
    text = re.sub(r"\s+", " ", text)
    return text.strip()