"""
OpenAI-compatible server for GuardReasoner-Omni-3B.
Supports text, image, and video moderation with chain-of-thought reasoning.
Uses HuggingFace Transformers + optional bitsandbytes 4-bit quantization.

Usage:
    # Full precision (bfloat16, needs ~12GB VRAM or ~12GB RAM on CPU)
    python guard_reasoner_server.py

    # 4-bit quantization (needs ~4GB VRAM, much faster)
    python guard_reasoner_server.py --quantize

    pip install fastapi uvicorn transformers torch pillow requests bitsandbytes accelerate qwen-omni-utils
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
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from qwen_omni_utils import process_mm_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration

MODEL_ID = "zhu-thu-22/GuardReasoner-Omni-3B"
PORT = 8001

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


def get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {name}  ({vram:.1f} GB VRAM)")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("GPU detected: Apple MPS")
        return "mps"
    print("No GPU found — loading on CPU (will be slow)")
    return "cpu"


def load_model(quantize: bool):
    device = get_device()
    print(f"Loading {MODEL_ID} {'(4-bit quantized)' if quantize else '(bfloat16)'} on {device}…")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    kwargs = dict(trust_remote_code=True, device_map=device)
    if quantize:
        if device not in ("cuda",):
            raise RuntimeError("4-bit quantization requires a CUDA GPU")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    model.disable_talker()
    print(f"Model ready on {device}.")
    return processor, model


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


# ── helpers ────────────────────────────────────────────────────────────────

def decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, data = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    resp = http_requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def save_video_to_temp(url: str) -> str:
    """Write video bytes to a named temp file and return its path. Caller must delete."""
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
    """Return (user_text, pil_images, video_path_or_None).
    video_path is a temp file — caller must delete it after inference.
    """
    texts = []
    images = []
    video_path = None

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
                    # video_url reuses image_url field to carry the data URL
                    video_path = save_video_to_temp(part.image_url.url)

    return " ".join(texts), images, video_path


def build_messages(user_text: str, images: list, video_path: str | None) -> list:
    """
    Build Qwen2.5-Omni message dicts for process_mm_info + apply_chat_template.
    Video uses the native {"type": "video"} content type so the model's temporal
    encoder is used — not individual image frames.
    """
    user_content = []

    if video_path:
        # Native video content type: process_mm_info will handle frame sampling
        user_content.append({
            "type": "video",
            "video": video_path,
            "fps": 1,
            "max_frames": 128,
            "min_pixels": 4 * 28 * 28,   # matches model training config
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
    """Extract structured verdict from <result>...</result> block."""
    result_match = re.search(r"<result>(.*?)</result>", raw, re.DOTALL | re.IGNORECASE)
    think_match  = re.search(r"<think>(.*?)</think>",   raw, re.DOTALL | re.IGNORECASE)

    result_text = result_match.group(1).strip() if result_match else raw.strip()
    reasoning   = think_match.group(1).strip()  if think_match  else ""

    cleaned = result_text.lower().replace("unharmful", "SAFE")
    is_harmful = "harmful" in cleaned

    verdict = "unsafe" if is_harmful else "safe"
    return {"verdict": verdict, "result": result_text, "reasoning": reasoning}


# ── app setup ──────────────────────────────────────────────────────────────

def make_app(processor, model):
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

        prompt_tokens = completion_tokens = total_tokens = 0
        try:
            text_input = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # process_mm_info uses Qwen's native pipeline:
            # - for images: loads PIL images
            # - for video: samples frames at specified fps with temporal encoding
            _, proc_images, proc_videos = process_mm_info(messages, use_audio_in_video=False)

            inputs = processor(
                text=text_input,
                images=proc_images if proc_images else None,
                videos=proc_videos if proc_videos else None,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    generation_mode="text",
                    thinker_max_new_tokens=req.max_tokens or 512,
                    do_sample=False,
                )

            new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
            raw_output = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

            # Save token counts before tensors are freed
            prompt_tokens     = int(inputs["input_ids"].shape[-1])
            completion_tokens = int(new_tokens.shape[-1])
            total_tokens      = int(output_ids.shape[-1])

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
                "total_tokens":      total_tokens,
            },
        }

    return app


# ── entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action="store_true",
                        help="Load in 4-bit with bitsandbytes (saves ~8GB memory)")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    processor, model = load_model(args.quantize)
    app = make_app(processor, model)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
