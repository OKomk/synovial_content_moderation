"""
Download GuardReasoner-Omni evaluation data from HuggingFace.

Media sizes (test split):
  mm_data/test/images.tar.gz   ~5.5 GB
  mm_data/test/audios.tar.gz   ~3.1 GB
  mm_data/test/videos.tar.gz  ~22.6 GB
"""

import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

REPO = "zhu-thu-22/GuardReasoner-Omni-data"
OUT  = Path("Content-moderation-dataset/GuardReasoner-Omni-data")

# ── Step 1: label/JSON files (fast, ~few MB) ──────────────────────────────────
print("Downloading label files…")
snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    local_dir=OUT,
    ignore_patterns=["mm_data/*"],
)
print("Labels done.\n")

# ── Step 2: test media — uncomment what you need ──────────────────────────────

MEDIA = [
    "mm_data/test/images.tar.gz",   # ~5.5 GB
    "mm_data/test/audios.tar.gz", # ~3.1 GB
    "mm_data/test/videos.tar.gz", # ~22.6 GB
]

for path in MEDIA:
    dest = OUT / path
    if dest.exists():
        print(f"Already downloaded: {path}")
    else:
        print(f"Downloading {path}…")
        hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=path,
            local_dir=OUT,
        )
        print(f"Saved to {dest}")

    # Extract
    extract_dir = dest.parent / dest.name.replace(".tar.gz", "")
    if not extract_dir.exists():
        print(f"Extracting to {extract_dir}…")
        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(extract_dir)
        print("Extracted.")
    else:
        print(f"Already extracted: {extract_dir}")
