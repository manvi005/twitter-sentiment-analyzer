import re
import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_URL  = "https://huggingface.co/mew205/twitter-sentiment-roberta"
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"}

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

def preprocess(text: str) -> str:
    tokens = []
    for token in text.split():
        if token.startswith("@") and len(token) > 1:
            token = "@user"
        elif token.startswith("http"):
            token = "http"
        tokens.append(token)
    return " ".join(tokens)

def predict(text: str) -> dict:
    cleaned = preprocess(text)

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": cleaned},
            timeout=30
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise Exception("HuggingFace API timed out. Try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"HuggingFace API error: {str(e)}")

    results = response.json()

    # Handle model loading response
    if isinstance(results, dict) and "error" in results:
        raise Exception(f"Model error: {results['error']}")

    mapped = [
        {
            "label": LABEL_MAP.get(r["label"], r["label"]),
            "score": r["score"]
        }
        for r in results[0]
    ]

    top = max(mapped, key=lambda x: x["score"])

    return {
        "text":       text,
        "sentiment":  top["label"],
        "confidence": round(top["score"], 4),
        "scores": {
            r["label"]: round(r["score"], 4)
            for r in mapped
        }
    }

def batch_predict(texts: list) -> list:
    return [predict(t) for t in texts]