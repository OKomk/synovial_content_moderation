"""
OpenAI-compatible server for GuardReasoner-Omni-3B.
Supports text, image, and video moderation with chain-of-thought reasoning.
Uses HuggingFace Transformers + optional bitsandbytes 4-bit quantization.

Usage:
    # Full precision (bfloat16, needs ~12GB VRAM or ~12GB RAM on CPU)
    python guard_reasoner_server.py

    # 4-bit quantization (needs ~4GB VRAM, much faster)
    python guard_reasoner_server.py --quantize

    pip install fastapi uvicorn transformers torch pillow requests bitsandbytes accelerate
"""

import argparse
import base64
import io
import re
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
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration

MODEL_ID = "zhu-thu-22/GuardReasoner-Omni-3B"
PORT = 8001  # different port from llama guard server

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


def load_model(quantize: bool):
    print(f"Loading {MODEL_ID} {'(4-bit quantized)' if quantize else '(bfloat16)'}…")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
    )
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    model.disable_talker()  # we only need text output, not TTS
    print("Model ready.")
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
    max_tokens: Optional[int] = 512   # needs more tokens for chain-of-thought


# ── helpers ────────────────────────────────────────────────────────────────

def decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, data = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    resp = http_requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def extract_text_and_images(messages: List[Message]):
    """Pull out the text content and any images from the request messages."""
    texts = []
    images = []
    for msg in messages:
        if isinstance(msg.content, str):
            texts.append(msg.content)
        else:
            for part in msg.content:
                if part.type == "text" and part.text:
                    texts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    images.append(decode_image(part.image_url.url))
    return " ".join(texts), images


def build_messages(user_text: str, images: list) -> list:
    """
    GuardReasoner expects:
      system: INSTRUCTION
      user:   [optional image(s)] "Human User:\n{text}\n\nAI assistant:\n"
    We treat the incoming content as the thing to moderate (the "Human User" turn).
    """
    user_content = []

    # Add images before the text (Qwen2.5-Omni style)
    for img in images:
        user_content.append({"type": "image", "image": img})

    # Wrap the content in the GuardReasoner prompt format
    user_content.append({
        "type": "text",
        "text": f"Human User:\n{user_text}\n\nAI assistant:\n[no response — assess the user request only]"
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

    # Classify as harmful if either the request or response is flagged
    is_harmful = "harmful" in result_text.lower() and "unharmful" not in result_text.lower().replace("harmful", "")
    # More careful: check if "harmful" appears outside of "unharmful"
    cleaned = result_text.lower().replace("unharmful", "SAFE")
    is_harmful = "harmful" in cleaned

    verdict = "unsafe" if is_harmful else "safe"
    return {"verdict": verdict, "result": result_text, "reasoning": reasoning}


# ── app setup (done after model loads) ─────────────────────────────────────

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
        user_text, images = extract_text_and_images(req.messages)
        messages = build_messages(user_text, images)

        print(f"[INFO] images={len(images)}  text_len={len(user_text)}")

        # Apply Qwen2.5-Omni chat template
        text_input = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs — images go through processor vision encoder
        inputs = processor(
            text=text_input,
            images=images if images else None,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                generation_mode="text",       # disable TTS talker
                thinker_max_new_tokens=req.max_tokens or 512,
                do_sample=False,
            )

        new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
        raw_output = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        parsed = parse_verdict(raw_output)
        print(f"[INFO] verdict={parsed['verdict']}  raw={raw_output[:120]}…")

        # Return both the structured verdict and full reasoning in the content
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
                "prompt_tokens":     int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(new_tokens.shape[-1]),
                "total_tokens":      int(output_ids.shape[-1]),
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
