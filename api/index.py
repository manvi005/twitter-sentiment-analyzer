from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from model import predict, batch_predict

app = FastAPI(title="Twitter Sentiment Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static files (Lottie JSON animations) ────────────────────
app.mount(
    "/static",
    StaticFiles(
        directory=os.path.join(os.path.dirname(__file__), "../frontend/static")
    ),
    name="static"
)

class SingleInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]

@app.get("/")
def root():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "../frontend/index.html")
    )

@app.get("/health")
def health():
    return {"status": "healthy", "model": "twitter-roberta-base-sentiment-latest"}

@app.post("/analyze")
def analyze(body: SingleInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(body.text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 chars)")
    try:
        return predict(body.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
def batch(body: BatchInput):
    if not body.texts:
        raise HTTPException(status_code=400, detail="texts list is empty")
    if len(body.texts) > 20:
        raise HTTPException(status_code=400, detail="Max 20 texts per batch")
    try:
        return {"results": batch_predict(body.texts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))