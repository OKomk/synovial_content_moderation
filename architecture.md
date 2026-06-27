# Content Moderation — Model Architecture

Exploring a tiered approach: a lightweight fast model for high-volume pre-screening, with a heavier reasoning model for borderline or high-stakes cases.

---

## Models Under Evaluation

### Llama Guard 4-12B — *primary (text + image)*
- **By:** Meta
- **Size:** 12B params (~14GB bfloat16)
- **Modalities:** Text, image (single image per request)
- **Safety categories:** S1–S14 (violent crimes, sexual content, self-harm, etc.)
- **Architecture:** Densely pruned from Llama 4 Scout; uses Llama 4 vision encoder
- **Status:** ✅ Working locally via `server.py`. Text ~10s, image ~6 min on CPU
- **Strengths:** Strong category coverage, OpenAI-compatible API, HuggingFace native
- **Weaknesses:** Slow on CPU; missed illustrated NSFW (false negative on censored anime art); no video support
- **Deployment:** Needs GPU (16GB+ VRAM) for practical latency

---

### SafeWatch 2B / 3B — *candidate (video)*
- **By:** ByteDance / HKUST
- **Size:** 2B or 3B params
- **Modalities:** Video (frame-level + temporal reasoning)
- **Status:** 🔬 To be evaluated
- **Strengths:** Designed specifically for video content safety; temporal understanding across frames; much smaller than Llama Guard 4
- **Weaknesses:** Video-specific — not a drop-in for text/image moderation; less coverage of nuanced categories
- **Use case here:** Pre-screening video uploads before expensive full moderation

---

### GuardReasoner Omni — *candidate (multimodal reasoning)*
- **By:** GuardReasoner team
- **Size:** ~7B (TBC)
- **Modalities:** Text, image, audio, video
- **Status:** 🔬 To be evaluated
- **Strengths:** Chain-of-thought safety reasoning — explains *why* content is unsafe, not just a label; omni-modal (single model for all content types); strong on ambiguous/borderline cases
- **Weaknesses:** Larger inference cost due to reasoning tokens; slower than classifier-only models
- **Use case here:** Second-pass reasoning on borderline decisions from the fast-path model

---

### NudeNet / Small Specialized Models — *candidate (cost efficiency)*
- **Examples:** NudeNet, NSFW-MobileNet, NSFWJS (client-side), Falconsai NSFW detector
- **Size:** <100MB
- **Modalities:** Image only
- **Status:** 🔬 NSFWJS already prototyped (`content-checker-test.html`)
- **Strengths:** Runs in-browser or on CPU in milliseconds; near-zero cost at scale; great for visual nudity detection specifically
- **Weaknesses:** Narrow category coverage (nudity only, no text/context/violence); no category breakdown; brittle on illustrations/art
- **Use case here:** Fast pre-filter before invoking the 12B model — reject obvious cases cheaply

---

## Proposed Tiered Architecture

```
Incoming content
       │
       ▼
┌──────────────────────────┐
│  Tier 1: Fast filter     │  NudeNet / NSFWJS
│  <100ms, CPU/browser     │  → PASS/FAIL on visual nudity only
└──────────┬───────────────┘
           │ flagged or uncertain
           ▼
┌──────────────────────────┐
│  Tier 2: Full moderation │  Llama Guard 4-12B  (text + image)
│  ~1s on GPU              │  SafeWatch 2B/3B    (video)
└──────────┬───────────────┘
           │ borderline / high-stakes
           ▼
┌──────────────────────────┐
│  Tier 3: Reasoning       │  GuardReasoner Omni
│  explains decision       │  → human review queue with explanation
└──────────────────────────┘
```

Most traffic is handled cheaply at Tier 1. Tier 2 handles flagged content across all modalities. Tier 3 is reserved for ambiguous cases or where an audit trail is needed.

---

## GPU Requirements Summary

| Model | Min VRAM | Inference latency (est. GPU) |
|---|---|---|
| Llama Guard 4-12B | 16GB | ~1–2s / request |
| SafeWatch 2B | 6GB | <1s / video clip |
| SafeWatch 3B | 8GB | ~1s / video clip |
| GuardReasoner Omni | 16GB | ~3–5s (reasoning) |
| NudeNet / NSFWJS | None (CPU) | <100ms |

---

## Deployment Target

Adobe's **Colligo inference platform** — wrap each model as a `ColligoMLModel` worker, allocate GPU via `build.gpu_info(memory_gb=N)`, upload weights to S3 via `./m deps.upload_artifact`. See the Colligo docs at `docs.ai.corp.adobe.com/models/colligo`.

For **AI Foundry**: not suitable for custom model hosting — it's a managed catalog of pre-approved models only. Could request Llama Guard 4 be added via `aifoundry-preview.corp.adobe.com/byo-models` for production use.
