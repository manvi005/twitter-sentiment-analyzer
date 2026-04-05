# Twitter Sentiment Analyzer

Sentiment analysis on tweets using `cardiffnlp/twitter-roberta-base-sentiment-latest` and a custom fine-tuned model `mew205/twitter-sentiment-roberta`.

## Live Links

| | Link |
|---|---|
| Live Website | https://twitter-sentiment-analyzer.vercel.app |
| API Docs | https://twitter-sentiment-analyzer.vercel.app/docs |
| Health Check | https://twitter-sentiment-analyzer.vercel.app/health |
| HuggingFace Model | https://huggingface.co/mew205/twitter-sentiment-roberta |

## Tech Stack

- Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Fine-tuned: `mew205/twitter-sentiment-roberta`
- Dataset: `tweet_eval` (sentiment)
- Backend: FastAPI
- Deployment: Vercel
- Training: Kaggle (P100 GPU)

## Local Setup
```bash
# Clone the repo
git clone https://github.com/manvi005/twitter-sentiment-analyzer.git
cd twitter-sentiment-analyzer

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements-local.txt

# Add your HuggingFace token to .env
HF_TOKEN=hf_xxx
USE_LOCAL_MODEL=true

# Start the server
uvicorn api.index:app --reload
```

## API Usage
```bash
# Single prediction
curl -X POST https://twitter-sentiment-analyzer.vercel.app/analyze \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I love this product\"}"

# Response
{
  "text": "I love this product",
  "sentiment": "positive",
  "confidence": 0.9812,
  "scores": {
    "negative": 0.0091,
    "neutral": 0.0097,
    "positive": 0.9812
  }
}
```

## License
Model: CC-BY 4.0 · Dataset: CC-BY 3.0 + Twitter ToS