# 🌑 Moon POI Viewer — NASA-IBM Lunar Foundation Model

An interactive Python application that displays the near-side Moon, lets you
click named **hot zones** (maria, craters, mountain ranges, landing sites) to
encode the image patch with the NASA-IBM Lunar Foundation Model and search for
similar features, and also provides a **text search** bar backed by the same
semantic vector space — all stored in DataStax **Astra DB**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI  (app.py)                                     │
│                                                             │
│  ┌──────────────┐   click    ┌──────────────────────────┐   │
│  │  Moon image  │ ─────────▶ │ crop 224×224 image patch │   │
│  │  + hot zones │            └────────────┬─────────────┘   │
│  └──────────────┘                         │                 │
│                                           ▼                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  model.py  (NASA-IBM LFM ViT-B backbone)              │  │
│  │  embed_image() → 768-d L2-normalised vector           │  │
│  │  embed_text()  → MiniLM-384 → linear proj → 768-d     │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  astra_client.py  (astrapy ≥ 2.0)                     │  │
│  │  Collection: moon_features  (cosine, dim=768)         │  │
│  │  vector_search(), upsert_features(), get_feature()    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Model:** [nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model)  
A ViT-B encoder–decoder trained on ~2 million co-registered lunar tile bundles
(LROC NAC/WAC, topography, spectral data). The encoder produces 768-d
representations of lunar imagery.  Text queries are handled by
`all-MiniLM-L6-v2` with an orthogonal linear projection to 768-d so both
modalities share the same Astra DB collection.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 + | Tested with 3.11 |
| DataStax Astra DB | Free tier sufficient — create a database at [astra.datastax.com](https://astra.datastax.com) |
| ~4 GB disk | NASA-IBM LFM backbone checkpoint (≈ 350 MB) + model caches |
| Optional: GPU | CUDA or MPS will speed up embedding; CPU works fine for this demo |

---

## Quick Start

### 1 — Clone and install

```bash
git clone <your-repo>
cd moon-viewer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Configure credentials

```bash
cp .env.example .env
# edit .env and fill in ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN
```

Your Astra DB **API Endpoint** and **Application Token** are found in the
Astra DB Console → your database → Connect tab.

### 3 — Seed the database

```bash
python seed_data.py
```

This embeds 17 canonical lunar features (maria, craters, mountain ranges,
Apollo landing sites) and upserts them into the `moon_features` Astra DB
collection.  Re-running is safe — documents are replaced, not duplicated.

> **First run:** the NASA-IBM LFM backbone (~350 MB) and the MiniLM model
> (~90 MB) are downloaded from HuggingFace and cached in `~/.cache/huggingface`.
> Set `HUGGINGFACE_TOKEN` in `.env` only if the model repository requires
> authentication.

### 4 — Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Clicking hot zones

The annotated moon map shows coloured rectangles over named features:

| Colour | Feature type |
|---|---|
| Blue | Maria / Oceanus |
| Orange | Craters |
| Green | Mountain ranges |
| Yellow | Apollo landing sites |

Click any of the buttons beneath the map (or the zone directly when using
`streamlit-image-coordinates` — see **Extensions** below) to:

1. Crop the 700 × 700 moon image to that zone's bounding box.
2. Resize the crop to 224 × 224 and feed it through the LFM ViT-B encoder.
3. Query Astra DB for the 6 most similar features by cosine similarity.
4. Display result cards with name, coordinates, diameter, description, and
   similarity score.

### Text search

Type any descriptive query in the sidebar (e.g. *"dark basaltic volcanic plain"*,
*"Apollo 11 landing site"*, *"permanently shadowed ice deposit"*) and click
**Search**.  The text is embedded with MiniLM and projected to 768-d before the
same Astra DB cosine search is performed.

### Type filter

Use the **Filter by type** dropdown in the sidebar to restrict results to a
specific feature category.

---

## File Layout

```
moon-viewer/
├── app.py              # Streamlit UI — hot zones, text search, result cards
├── model.py            # NASA-IBM LFM image encoder + MiniLM text encoder
├── astra_client.py     # astrapy ≥ 2.0 wrapper (collection, search, upsert)
├── seed_data.py        # Populate Astra DB with canonical lunar features
├── requirements.txt
├── .env.example
├── assets/
│   └── moon.jpg        # Downloaded automatically on first run
└── data/               # Reserved for future datasets / caches
```

---

## Extending the Dataset

Add entries to the `FEATURES` list in [`seed_data.py`](seed_data.py) following
the same schema and re-run `python seed_data.py`.  Each document requires at
minimum `_id`, `name`, `type`, `description`, `lat`, `lon`.

To index **actual LROC image tiles** instead of text descriptions, call
`embed_image()` from `model.py` on each tile and include the vector in the
document's `"$vector"` field when upserting.

---

## Acknowledgements

- **NASA-IBM Lunar Foundation Model** — Fraccaro et al. (2026),
  [HuggingFace](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model),
  Apache-2.0 licence.
- **Moon image** — NASA / GSFC / Arizona State University (public domain).
- **Astra DB** — DataStax, [astra.datastax.com](https://astra.datastax.com).
- Lunar feature data — IAU Gazetteer of Planetary Nomenclature / USGS.
