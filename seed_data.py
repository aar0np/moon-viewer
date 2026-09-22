"""
seed_data_csv.py
----------------
Populate Astra DB with lunar features read from data/moon_poi.csv.

Each feature description is fed through embed_text() from model.py to produce
the 768-d embedding stored in the "$vector" field.

Usage:
    python seed_data_csv.py

The script is idempotent — re-running it refreshes/replaces the embeddings.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Make sure the project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

# Load .env BEFORE importing any project modules so environment variables
# (including HUGGINGFACE_TOKEN) are set before lazy model loaders read them.
from dotenv import load_dotenv
load_dotenv()

from model import embed_text
from astra_client import upsert_features, count_features

# ---------------------------------------------------------------------------
# CSV path
# ---------------------------------------------------------------------------

CSV_PATH = Path(__file__).parent / "data" / "moon_poi.csv"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_features(csv_path: Path) -> list[dict]:
    """Read moon_poi.csv and return a list of feature dicts."""
    features = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # diameter_km is empty for point features (landing sites, etc.)
            raw_diam = row["diameter_km"].strip()
            features.append({
                "_id":         row["_id"].strip(),
                "name":        row["name"].strip(),
                "type":        row["type"].strip(),
                "lat":         float(row["lat"]),
                "lon":         float(row["lon"]),
                "diameter_km": float(raw_diam) if raw_diam else None,
                "source":      row["source"].strip(),
                "description": row["description"].strip(),
            })
    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def seed():
    features = load_features(CSV_PATH)
    print(f"Loaded {len(features)} features from {CSV_PATH.relative_to(Path(__file__).parent)}")
    print(f"Embedding {len(features)} lunar features …")

    docs = []
    for i, feat in enumerate(features):
        text = f"{feat['name']}: {feat['description']}"
        vec = embed_text(text)
        doc = {**feat, "$vector": vec}
        docs.append(doc)
        print(f"  [{i + 1:2d}/{len(features)}] {feat['name']}")

    print("\nUpserting to Astra DB …")
    upsert_features(docs)
    total = count_features()
    print(f"Done — collection now contains {total} documents.")


if __name__ == "__main__":
    seed()
