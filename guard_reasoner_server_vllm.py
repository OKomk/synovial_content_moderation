"""
vLLM-based OpenAI-compatible server for GuardReasoner-Omni-3B.
Replaces HuggingFace generate() with vLLM's AsyncLLMEngine for:
  - PagedAttention (efficient KV cache memory)
  - Continuous batching (multiple concurrent requests share GPU compute)
  - CUDA graph optimisation (reduced Python overhead per token)

Expected speedup over HF server: 3-5x on A100.

Usage:
    pip install vllm qwen-omni-utils
    python guard_reasoner_server_vllm.py
    python guard_reasoner_server_vllm.py --quantize    # bitsandbytes 4-bit
    python guard_reasoner_server_vllm.py --port 8002   # run alongside HF server
"""

import argparse
import base64
import io
import os
import re
import tempfile
import time
import uuid
from typing import List, Optional, Union

import requests as http_requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from qwen_omni_utils import process_mm_info
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

MODEL_ID = "zhu-thu-22/GuardReasoner-Omni-3B"
PORT = 8002   # run alongside the HF server on 8001

INSTRUCTION = (
    "You are a classifier for analyzing interactions between humans and AI. "
    "Given a request from a human user and a response from an AI assistant, "
    "finish two tasks. "
    "Task 1: determine whether the human user's request is harmful or unharmful. "
    "Task 2: determine whether the AI assistant's response is harmful or unharmful. "
    "Think step-by-step. Put your reasoning in <think></think> tags and your "
    "final results in <result></result> tags. "
    "In the <result> block write exactly: "
    "'User request: harmful/unharmful. AI response: harmful/unharmful.'"
)


def load_engine(quantize: bool) -> AsyncLLMEngine:
    print(f"Loading {MODEL_ID} with vLLM {'(4-bit bitsandbytes)' if quantize else '(bfloat16)'}…")

    kwargs = dict(
        model=MODEL_ID,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 16, "video": 4},
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    if quantize:
        kwargs["quantization"] = "bitsandbytes"
        kwargs["load_format"] = "bitsandbytes"

    engine_args = AsyncEngineArgs(**kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("vLLM engine ready.")
    return engine


# ── request schema (OpenAI-compatible subset) ──────────────────────────────

class ImageUrl(BaseModel):
    url: str

class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]

class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[Message]
    max_tokens: Optional[int] = 512


# ── helpers (identical to HF server) ───────────────────────────────────────

def decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, data = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    resp = http_requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def save_video_to_temp(url: str) -> str:
    if url.startswith("data:"):
        _, data = url.split(",", 1)
        raw = base64.b64decode(data)
    else:
        raw = http_requests.get(url, timeout=30).content
    suffix = ".mp4"
    if url.startswith("data:video/"):
        mime = url.split(";")[0].split("/")[1]
        suffix = f".{mime}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        return f.name


def extract_content(messages: List[Message]):
    texts, images, video_path = [], [], None
    for msg in messages:
        if isinstance(msg.content, str):
            texts.append(msg.content)
        else:
            for part in msg.content:
                if part.type == "text" and part.text:
                    texts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    images.append(decode_image(part.image_url.url))
                elif part.type == "video_url" and part.image_url:
                    video_path = save_video_to_temp(part.image_url.url)
    return " ".join(texts), images, video_path


def build_messages(user_text: str, images: list, video_path: str | None) -> list:
    user_content = []
    if video_path:
        user_content.append({
            "type": "video",
            "video": video_path,
            "fps": 1,
            "max_frames": 128,
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 64 * 28 * 28,
        })
    else:
        for img in images:
            user_content.append({"type": "image", "image": img})
    user_content.append({
        "type": "text",
        "text": f"Human User:\n{user_text}\n\nAI assistant:\n[no response — assess the user request only]",
    })
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user",   "content": user_content},
    ]


def parse_verdict(raw: str) -> dict:
    result_match = re.search(r"<result>(.*?)</result>", raw, re.DOTALL | re.IGNORECASE)
    think_match  = re.search(r"<think>(.*?)</think>",   raw, re.DOTALL | re.IGNORECASE)
    result_text = result_match.group(1).strip() if result_match else raw.strip()
    reasoning   = think_match.group(1).strip()  if think_match  else ""
    cleaned = result_text.lower().replace("unharmful", "SAFE")
    is_harmful = "harmful" in cleaned
    return {"verdict": "unsafe" if is_harmful else "safe", "result": result_text, "reasoning": reasoning}


# ── app ─────────────────────────────────────────────────────────────────────

def make_app(engine: AsyncLLMEngine, processor) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        user_text, images, video_path = extract_content(req.messages)
        messages = build_messages(user_text, images, video_path)

        modality = "video" if video_path else (f"{len(images)} image(s)" if images else "text-only")
        print(f"[INFO] modality={modality}  text_len={len(user_text)}")

        try:
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            _, proc_images, proc_videos = process_mm_info(messages, use_audio_in_video=False)

            mm_data = {}
            if proc_images:
                mm_data["image"] = proc_images
            if proc_videos:
                mm_data["video"] = proc_videos

            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=req.max_tokens or 512,
            )

            # vLLM multimodal input format: dict with "prompt" + optional "multi_modal_data"
            vllm_input: dict = {"prompt": text_input}
            if mm_data:
                vllm_input["multi_modal_data"] = mm_data

            request_id = uuid.uuid4().hex
            results_gen = engine.generate(
                vllm_input,
                sampling_params,
                request_id=request_id,
            )

            # Collect full output (non-streaming)
            final_output = None
            async for out in results_gen:
                final_output = out

            raw_output = final_output.outputs[0].text.strip()
            prompt_tokens     = len(final_output.prompt_token_ids)
            completion_tokens = len(final_output.outputs[0].token_ids)

        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)

        parsed = parse_verdict(raw_output)
        print(f"[INFO] verdict={parsed['verdict']}  raw={raw_output[:120]}…")

        content = f"{parsed['verdict']}\n\n{parsed['result']}\n\n<reasoning>{parsed['reasoning']}</reasoning>"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens":      prompt_tokens + completion_tokens,
            },
        }

    return app


# ── entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    # vLLM engine
    engine = load_engine(args.quantize)

    # Processor for chat template + process_mm_info (no model weights needed)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    app = make_app(engine, processor)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
