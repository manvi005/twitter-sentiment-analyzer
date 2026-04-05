import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN       = os.environ.get("HF_TOKEN", "")
USE_LOCAL      = os.environ.get("USE_LOCAL_MODEL", "false").lower() == "true"
LOCAL_MODEL_ID = "mew205/twitter-sentiment-roberta"
API_MODEL_ID   = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# ── Load the right backend once at startup ────────────────────────────
if USE_LOCAL:
    from transformers import pipeline
    print(f"Loading local model: {LOCAL_MODEL_ID}")
    print("First run downloads ~500MB — subsequent runs are instant...")
    _pipe = pipeline(
        task="text-classification",
        model=LOCAL_MODEL_ID,
        top_k=None,
        device=-1        # CPU (your GPU has too little VRAM)
    )
    print("Local model ready.")
else:
    from huggingface_hub import InferenceClient
    print(f"Using HuggingFace Inference API: {API_MODEL_ID}")
    _client = InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN,
    )

# ── Shared preprocessing ──────────────────────────────────────────────
def preprocess(text: str) -> str:
    tokens = []
    for token in text.split():
        if token.startswith("@") and len(token) > 1:
            token = "@user"
        elif token.startswith("http"):
            token = "http"
        tokens.append(token)
    return " ".join(tokens)

# ── Local inference ───────────────────────────────────────────────────
def _predict_local(text: str) -> dict:
    cleaned = preprocess(text)
    results = _pipe(cleaned)[0]

    mapped = [
        {
            "label": LABEL_MAP.get(r["label"], r["label"]),
            "score": r["score"]
        }
        for r in results
    ]

    top = max(mapped, key=lambda x: x["score"])

    return {
        "text":       text,
        "sentiment":  top["label"],
        "confidence": round(top["score"], 4),
        "scores":     {r["label"]: round(r["score"], 4) for r in mapped},
        "model":      LOCAL_MODEL_ID,
        "mode":       "local"
    }

# ── API inference ─────────────────────────────────────────────────────
def _predict_api(text: str) -> dict:
    cleaned = preprocess(text)

    try:
        results = _client.text_classification(
            text=cleaned,
            model=API_MODEL_ID,
        )
    except Exception as e:
        raise Exception(f"Inference error: {str(e)}")

    mapped = [
        {
            "label": LABEL_MAP.get(r.label, r.label),
            "score": r.score
        }
        for r in results
    ]

    top = max(mapped, key=lambda x: x["score"])

    return {
        "text":       text,
        "sentiment":  top["label"],
        "confidence": round(top["score"], 4),
        "scores":     {r["label"]: round(r["score"], 4) for r in mapped},
        "model":      API_MODEL_ID,
        "mode":       "api"
    }

# ── Public interface ──────────────────────────────────────────────────
def predict(text: str) -> dict:
    if USE_LOCAL:
        return _predict_local(text)
    else:
        return _predict_api(text)

def batch_predict(texts: list) -> list:
    return [predict(t) for t in texts]