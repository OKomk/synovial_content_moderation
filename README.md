# Content Moderation — GarageWeek 2026

Exploring content safety models for text and image/video moderation. Running models locally to benchmark accuracy, latency, and cost before committing to a production deployment path.

## What's here

| File | Description |
|---|---|
| `server.py` | OpenAI-compatible FastAPI server wrapping Llama Guard 4-12B (HuggingFace Transformers, CPU) |
| `llama-guard-test.html` | Browser UI for testing the server — text + image inputs |
| `content-checker-test.html` | Earlier browser-based NSFW check using nsfwjs (client-side TensorFlow.js) |
| `Content-moderation-dataset/` | Sample images for manual testing |

## Quickstart

```bash
# Install deps (uses the POC conda env)
conda activate POC
pip install fastapi uvicorn transformers torch pillow requests

# Start the server (loads ~14GB model into RAM, takes ~20s)
python server.py

# Open the test UI
open llama-guard-test.html
```

Server runs at `http://localhost:8000`. API is OpenAI-compatible (`/v1/chat/completions`).

## Known issues

- Image inference takes ~6 min on CPU — needs GPU for practical use
- `attention_chunk_size` and `use_cache` patches required for transformers 5.12+ compatibility with Llama Guard 4 config

## Deployment

See [`architecture.md`](architecture.md) for model comparison and the planned deployment path via Adobe's Colligo inference platform.
