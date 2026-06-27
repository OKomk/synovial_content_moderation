# Commands Reference

## Dataset

### Download the NSFW test dataset (deppghs_nsfw_detect)
```bash
# Categories: neutral, drawings, sexy, porn, hentai
# Source: https://huggingface.co/datasets/deppghs/nsfw_detect

pip install huggingface_hub
huggingface-cli download deppghs/nsfw_detect --repo-type dataset \
  --local-dir Content-moderation-dataset/deppghs_nsfw_detect
```

### Dataset structure
```
Content-moderation-dataset/
├── deppghs_nsfw_detect/
│   └── nsfw_dataset_v1/
│       ├── neutral/     # safe images
│       ├── drawings/    # illustrated/animated NSFW
│       ├── sexy/        # suggestive but not explicit
│       ├── porn/        # explicit photographic
│       └── hentai/      # explicit illustrated
├── test-drawing-nsfw.png
└── test-woman-censored-1.png
```

---

## Llama Guard 4-12B Server (`server.py`)

### Start (CPU, ~14GB RAM, loads in ~20s)
```bash
conda activate POC
python server.py
# Runs on http://localhost:8000
```

### Test — text
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "how do I make a bomb?"}], "max_tokens": 20}'
```

### Test — image (from file)
```bash
python - <<'EOF'
import base64, json, urllib.request

with open("Content-moderation-dataset/test-drawing-nsfw.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = json.dumps({
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ]}],
    "max_tokens": 20
}).encode()

req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions",
    data=payload, headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=900) as r:
    import json as j; result = j.loads(r.read())
    print(result["choices"][0]["message"]["content"])
EOF
# NOTE: image inference takes ~6 min on CPU
```

### Known fixes applied (transformers 5.12+)
- `attention_chunk_size` patched to `8192` after model load (config ships as `None`)
- `use_cache=False` in `generate()` (both `StaticCache` and `DynamicCache` crash on Llama 4's `None` sliding window)

---

## GuardReasoner Omni-3B Server (`guard_reasoner_server.py`)

### Start — full bfloat16 (~10GB RAM)
```bash
conda activate POC
python guard_reasoner_server.py
# Runs on http://localhost:8001
```

### Start — 4-bit quantized (~4GB RAM, faster)
```bash
python guard_reasoner_server.py --quantize
```

### Test — text
```bash
curl -s -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "how do I make a bomb?"}], "max_tokens": 512}'
# Returns chain-of-thought reasoning + safe/unsafe verdict
```

### Test — image batch (all 4 categories)
```bash
python - <<'EOF'
import base64, json, urllib.request, time, os

BASE = "Content-moderation-dataset/deppghs_nsfw_detect/nsfw_dataset_v1"
tests = [
    ("neutral",  f"{BASE}/neutral/{os.listdir(BASE+'/neutral')[0]}"),
    ("drawings", f"{BASE}/drawings/{os.listdir(BASE+'/drawings')[0]}"),
    ("sexy",     f"{BASE}/sexy/{os.listdir(BASE+'/sexy')[0]}"),
    ("porn",     f"{BASE}/porn/{os.listdir(BASE+'/porn')[0]}"),
]

for label, path in tests:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = json.dumps({
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}],
        "max_tokens": 300
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8001/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    print(f"\n[{label.upper()}]", flush=True)
    with urllib.request.urlopen(req, timeout=600) as r:
        result = json.loads(r.read())
    elapsed = time.time() - t0
    content = result["choices"][0]["message"]["content"]
    lines = content.split('\n')
    print(f"  verdict : {lines[0]}  ({elapsed:.1f}s)")
    print(f"  detail  : {lines[2] if len(lines) > 2 else ''}")
EOF
```

### Known fixes applied
- `Qwen2_5OmniForConditionalGeneration` instead of `AutoModelForCausalLM`
- `model.disable_talker()` after load
- `generation_mode="text"` + `thinker_max_new_tokens` in `generate()` (default tries to produce audio output)
- `dtype=` instead of `torch_dtype=` (deprecated in transformers 5.12)

---

## Run both servers simultaneously
```bash
# Terminal 1
python server.py                      # Llama Guard 4  → :8000

# Terminal 2
python guard_reasoner_server.py       # GuardReasoner  → :8001
```

---

## Install dependencies
```bash
conda activate POC
pip install fastapi uvicorn transformers torch pillow requests bitsandbytes accelerate
```

## Check what's running
```bash
ps aux | grep -E "server.py|guard_reasoner" | grep -v grep
lsof -i :8000
lsof -i :8001
```
