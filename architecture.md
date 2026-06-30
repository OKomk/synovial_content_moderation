# Content Moderation — Architecture

---

## Models

### GuardReasoner Omni-3B ✅ *primary (all modalities)*
- **Base:** Qwen2.5-Omni-3B, fine-tuned via SFT + GRPO on 148K safety samples
- **Modalities:** Text, image, video, audio
- **Safety categories:** Sexual content, violence, harassment, illegal activity, extremism, misinformation, hate speech
- **Training data (video):** SafeWatch-Bench (16K samples — sexual, violence, extremism, illegal), UCF-Crime, XD-Violence, Video-SafetyBench
- **Output:** Chain-of-thought reasoning in `<think>` tags + structured verdict in `<result>` tags
- **Status:** Running on A100 via Colligo. 4-bit quantized fits on RTX 4060 (8GB).
- **Latency (A100):** TBD — run `benchmark.py`

### Llama Guard 4-12B ⚠️ *text + image only*
- **Base:** Llama 4 Scout (12B), Meta
- **Modalities:** Text, image (no video)
- **Safety categories:** S1–S14
- **Status:** Working. CPU-only (no GPU access during GarageWeek). Text ~10s, image ~6 min on CPU.
- **Note:** Missed illustrated NSFW (false negative on censored anime art). Needs 16GB+ VRAM for practical latency.

### NudeNet / NSFWJS *fast pre-filter (image only)*
- **Modalities:** Image only
- **Status:** NSFWJS prototyped client-side in `content-checker-test.html`
- **Latency:** <100ms, runs in-browser

---

## Pipeline

```
Incoming content (text / image / video / audio)
        │
        ▼
┌─────────────────────────────────┐
│  Fast pass                      │  GuardReasoner Omni — low max_tokens (~128)
│  ~Xs on GPU                     │  Verdict only, no reasoning
└──────────┬──────────────────────┘
           │ unsafe OR confidence < threshold
           ▼
┌─────────────────────────────────┐
│  Slow pass                      │  GuardReasoner Omni — full reasoning (~512 tokens)
│  ~Xs on GPU                     │  Returns chain-of-thought explanation
└──────────┬──────────────────────┘
           │ unsafe
           ▼
   Human review queue
   (with CoT explanation attached)
```

> Fast/slow pass use the **same model** — fast pass reduces `max_tokens` to skip chain-of-thought
> generation and just get the verdict. Slow pass only runs on flagged content.
> Ensemble with a separate lightweight model (NudeNet for images) is a further option if
> fast-pass latency is still too high at scale.

---

## AEM Integration

### Option A: NUI Worker (preferred for GarageWeek)

Create a new FMT in NUI core that wraps the GuardReasoner Omni Colligo endpoint:

```
AEM asset upload / rendition event
        │
        ▼
  NUI Core (new FMT: content-moderation)
        │  calls Colligo worker via HTTP
        ▼
  GuardReasoner Omni (Colligo)
        │  returns verdict + reasoning
        ▼
  NUI writes result back to asset metadata
  (e.g. dam:moderationVerdict, dam:moderationReason)
```

- **FMT definition:** Add a new worker type in NUI core that accepts image/video renditions
- **Worker:** Colligo `ColligoMLModel` wrapping `guard_reasoner_server.py`
- **Output:** Write verdict + CoT back to AEM asset metadata via NUI result handler
- **Trigger:** On asset upload, or on-demand via workflow

### Option B: AEM Workflow Step (heavier, not needed for POC)

Direct Java OSGi workflow step calling the Colligo endpoint — more AEM-native but more
overhead to set up. Better for production where you need retry logic, SLA tracking, etc.

---

## Deployment (Colligo)

```python
# Colligo worker skeleton
from colligo import ColligoMLModel, build

class ContentModerationWorker(ColligoMLModel):
    @build.gpu_info(memory_gb=24)   # A10G or better
    def build(self): ...
    def predict(self, inputs): ...
```

Upload weights: `./m deps.upload_artifact`
Docs: `docs.ai.corp.adobe.com/models/colligo`

---

## Latency (to be filled after benchmark)

Run: `python benchmark.py --endpoint <colligo-or-ngrok-url>`

Measured on A100 via ngrok (add ~1-2s for ngrok overhead):

| Modality | Input size | Latency | Notes |
|---|---|---|---|
| Text | any | ~8.5s | Bottleneck is CoT generation, not input length |
| Image | 62 KB | ~26s | Vision encoder dominates |
| Video | 0.1 MB (~1s clip) | ~19s | Minimum video overhead |
| Video | 4.4 MB (~17s clip) | ~45s | Scales with frames sampled |

### Fast-pass (reduced max_tokens)

Cutting `max_tokens` reduces latency roughly linearly, but the model generates reasoning
*before* the verdict — so truncated output may not contain a usable verdict:

| max_tokens | Latency | Usable verdict? |
|---|---|---|
| 64 | ~3.3s | ❌ CoT truncated, no `<result>` block |
| 128 | ~5.2s | ⚠️ Sometimes truncated |
| 512 | ~8.7s | ✅ Full response (~214 tokens actual) |

**Experiment — verdict-first prompt:** Flipping the system prompt to output verdict before
reasoning was tested. Result: model produced wrong verdicts at low token counts (returned
"safe" for "how do I make a bomb?" at 64 tokens). The CoT reasoning IS the source of
accuracy — the model needs to think before it can reliably classify. ❌ Abandoned.

**Conclusion:** Fast pass via token truncation is not viable for this model. The right
optimisation path is vLLM (3-5x faster generation) + streaming (verdict feels instant
even if total time is the same). A separate lightweight gating model (NudeNet for images)
remains an option for obvious cases at scale.

---

## GPU Requirements

| Model | Min VRAM | Config |
|---|---|---|
| GuardReasoner Omni-3B bfloat16 | 8GB | Full precision |
| GuardReasoner Omni-3B 4-bit | 4GB | `--quantize` flag |
| Llama Guard 4-12B bfloat16 | 16GB | No quantization support |
