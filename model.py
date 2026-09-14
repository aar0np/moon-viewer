"""
model.py
--------
Lightweight wrapper around the NASA-IBM Lunar Foundation Model's ViT-B backbone
for producing 768-dimensional feature embeddings from an image patch or from a
short text description (via the same token-space used by the backbone).

The LFM is a *vision* foundation model (not a vision-language model), so text
queries are handled with a fallback: a lightweight sentence-transformers
all-MiniLM-L6-v2 model produces 384-d text embeddings that are linearly
projected up to 768-d to share the same Astra DB collection as image embeddings.
Both paths produce L2-normalised 768-d vectors — cosine similarity via
dot-product then works for cross-modal retrieval.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# NASA-IBM LFM backbone (ViT-B, 768 dim)
# ---------------------------------------------------------------------------

MODEL_REPO = "nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model"
EMBED_DIM = 768
TEXT_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dim
_CACHE_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


@lru_cache(maxsize=1)
def _load_vision_model() -> tuple[nn.Module, object]:
    """
    Download (once) and load the LFM backbone + its image preprocessor.

    Returns (model, transform).  The backbone is used in eval mode; grad
    computation is disabled at call time in embed_image().
    """
    from huggingface_hub import snapshot_download
    from timm.models.vision_transformer import VisionTransformer
    import timm

    # huggingface_hub reads HF_TOKEN from the environment automatically.
    # Bridge HUGGINGFACE_TOKEN (our .env key) → HF_TOKEN so the library
    # suppresses its "please set HF_TOKEN" warning and uses higher rate limits.
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token

    # Download just the backbone weights (≈ 350 MB, not the full 12 GB repo)
    local = snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=["backbone/*"],
        local_dir=str(_CACHE_DIR / "nasa-ibm-lfm"),
        token=hf_token,
    )
    ckpt_path = Path(local) / "backbone" / "checkpoint.pt"

    # Build a standard ViT-B/16 from timm and load the LFM weights.
    # The LFM encoder is ViT-B: embed_dim=768, depth=12, num_heads=12
    model: VisionTransformer = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,  # remove classification head → output is [B, 768]
    )
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    # The checkpoint may wrap weights under an "encoder" key
    weights = state.get("encoder", state.get("model", state))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        print(f"[model] ViT-B: {len(missing)} missing keys (expected for head removal)")
    model.eval()

    # Standard timm data config for ViT-B/16
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    return model, transform


@lru_cache(maxsize=1)
def _load_text_model() -> tuple[nn.Module, object, nn.Module]:
    """
    Load all-MiniLM-L6-v2 for text → 384-d embedding, plus a learned linear
    projection to 768-d so text and image embeddings share one Astra collection.

    Returns (sentence_model, tokenizer, projector).
    """
    from transformers import AutoTokenizer, AutoModel

    name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(name)
    text_model = AutoModel.from_pretrained(name)
    text_model.eval()

    # Deterministic linear projection (no training needed for retrieval demo)
    torch.manual_seed(42)
    projector = nn.Linear(TEXT_EMBED_DIM, EMBED_DIM, bias=False)
    nn.init.orthogonal_(projector.weight)
    projector.eval()

    return text_model, tokenizer, projector


# ---------------------------------------------------------------------------
# Public embedding functions
# ---------------------------------------------------------------------------

def embed_image(image: Image.Image) -> list[float]:
    """Return a 768-d L2-normalised embedding for the given PIL image."""
    model, transform = _load_vision_model()
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        vec = model(tensor)  # [1, 768]
    vec = torch.nn.functional.normalize(vec, dim=-1)
    return vec.squeeze(0).tolist()


def embed_text(text: str) -> list[float]:
    """Return a 768-d L2-normalised embedding for the given text string."""
    text_model, tokenizer, projector = _load_text_model()
    encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = text_model(**encoded)
        # mean-pool the last hidden state (standard sentence-transformers approach)
        hidden = out.last_hidden_state  # [1, seq_len, 384]
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [1, 384]
        projected = projector(pooled)  # [1, 768]
    projected = torch.nn.functional.normalize(projected, dim=-1)
    return projected.squeeze(0).tolist()
