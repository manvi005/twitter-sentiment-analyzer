from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../api"))
from index import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_analyze_valid():
    r = client.post("/analyze", json={"text": "I love this!"})
    print("\nRESPONSE:", r.json())  # add this line
    assert r.status_code == 200
    data = r.json()
    assert "sentiment"  in data
    assert "confidence" in data
    assert "scores"     in data
    assert data["sentiment"] in ["positive", "neutral", "negative"]

def test_analyze_empty():
    r = client.post("/analyze", json={"text": ""})
    assert r.status_code == 400

def test_batch():
    r = client.post("/batch", json={"texts": ["great!", "awful", "okay"]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3

def test_batch_too_large():
    r = client.post("/batch", json={"texts": ["test"] * 25})
    assert r.status_code == 400