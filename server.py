"""
Minimal OpenAI-compatible server for Llama Guard 4-12B.
Handles both text-only and image+text moderation correctly.
Uses HuggingFace Transformers directly (bypasses vLLM image bug).

Usage:
    pip install fastapi uvicorn transformers torch pillow requests
    python server.py
"""

import base64
import io
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
from transformers import AutoProcessor, Llama4ForConditionalGeneration

MODEL_ID = "meta-llama/Llama-Guard-4-12B"
PORT = 8000

print(f"Loading {MODEL_ID} …")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Llama4ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
# transformers 5.12+ requires attention_chunk_size; Llama Guard 4 config ships it as None
if not model.config.text_config.attention_chunk_size:
    model.config.text_config.attention_chunk_size = 8192
print("Model ready.")

# Safety categories per the Llama Guard 4 model card.
# Images exclude S14 (Code Interpreter Abuse — text-only category).
_CATS_TEXT = {
    "S1": "Violent Crimes.", "S2": "Non-Violent Crimes.", "S3": "Sex Crimes.",
    "S4": "Child Exploitation.", "S5": "Defamation.", "S6": "Specialized Advice.",
    "S7": "Privacy.", "S8": "Intellectual Property.", "S9": "Indiscriminate Weapons.",
    "S10": "Hate.", "S11": "Self-Harm.", "S12": "Sexual Content.",
    "S13": "Elections.", "S14": "Code Interpreter Abuse.",
}
_CATS_IMG = {k: v for k, v in _CATS_TEXT.items() if k != "S14"}


def build_llama_guard_prompt(hf_msgs: list, has_images: bool) -> str:
    """Build the Llama Guard 4 safety assessment prompt directly."""
    # Parity rule from the official template: odd-length → last is User, even → Agent
    role = "User" if len(hf_msgs) % 2 == 1 else "Agent"
    cats = _CATS_IMG if has_images else _CATS_TEXT

    parts = [
        "<|begin_of_text|><|header_start|>user<|header_end|>\n",
        f"Task: Check if there is unsafe content in '{role}' messages in conversations"
        " according to our safety policy with the below categories.\n",
        "<BEGIN UNSAFE CONTENT CATEGORIES>\n",
        *[f"{k}: {v}\n" for k, v in cats.items()],
        "<END UNSAFE CONTENT CATEGORIES>\n",
        "<BEGIN CONVERSATION>\n\n",
    ]

    for msg in hf_msgs:
        msg_role = "User" if msg["role"] == "user" else "Agent"
        img_prefix = ""
        texts = []
        for part in msg["content"]:
            if part["type"] == "image":
                img_prefix += "<|image|>"
            elif part["type"] == "text":
                texts.append(part["text"].strip())
        parts.append(f"{msg_role}: {img_prefix}{' '.join(texts)}\n\n")

    parts += [
        "<END CONVERSATION>\n",
        f"Provide your safety assessment for ONLY THE LAST {role} message in the above conversation:",
        " - First line must read 'safe' or 'unsafe'.",
        " - If unsafe, a second line must include a comma-separated list of violated categories."
        " <|eot|><|header_start|>assistant<|header_end|>",
    ]
    return "".join(parts)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    max_tokens: Optional[int] = 20


# ── helpers ────────────────────────────────────────────────────────────────

def decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, data = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    resp = http_requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def to_hf_messages(messages: List[Message]):
    """Convert OpenAI content parts → HuggingFace format + collect PIL images."""
    hf_msgs = []
    pil_images = []

    for msg in messages:
        if isinstance(msg.content, str):
            hf_msgs.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
            continue

        hf_content = []
        for part in msg.content:
            if part.type == "text" and part.text:
                hf_content.append({"type": "text", "text": part.text})
            elif part.type == "image_url" and part.image_url:
                img = decode_image(part.image_url.url)
                pil_images.append(img)
                hf_content.append({"type": "image"})   # HF uses placeholder, image passed separately

        hf_msgs.append({"role": msg.role, "content": hf_content})

    return hf_msgs, pil_images


# ── endpoint ───────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    hf_msgs, pil_images = to_hf_messages(req.messages)

    text = build_llama_guard_prompt(hf_msgs, has_images=bool(pil_images))
    print(f"[INFO] images={len(pil_images)}  <|image|> in prompt={text.count('<|image|>')}")

    inputs = processor(
        text=text,
        images=pil_images if pil_images else None,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens or 20,
            do_sample=False,
            use_cache=False,
        )

    new_tokens = outputs[:, inputs["input_ids"].shape[-1]:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "completion_tokens": int(new_tokens.shape[-1]),
            "total_tokens": int(outputs.shape[-1]),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
