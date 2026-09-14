"""
app.py
------
Moon Viewer — interactive Streamlit application.

Features:
  • Full-disk near-side moon image with clickable hot zones.
  • Each hot zone corresponds to a named lunar feature.  Clicking a zone crops
    that region, encodes it with the NASA-IBM Lunar Foundation Model (ViT-B
    backbone → 768-d embedding), and queries Astra DB for the most similar
    features.
  • A text search bar embeds the query text (384-d MiniLM → projected 768-d)
    and retrieves the closest matching features from Astra DB.
  • Results are displayed as expandable cards with name, type, coordinates,
    diameter, description, and similarity score.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

# Ensure local modules are importable when launching via `streamlit run`
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Moon Viewer — NASA-IBM LFM",
    page_icon="🌑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports (heavy ML libs loaded on first use so the UI renders fast)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading NASA-IBM Lunar Foundation Model …")
def _get_embed_image():
    from model import embed_image
    return embed_image


@st.cache_resource(show_spinner="Loading text embedding model …")
def _get_embed_text():
    from model import embed_text
    return embed_text


# ---------------------------------------------------------------------------
# Moon image — download NASA's full-disk near-side mosaic if not cached
# ---------------------------------------------------------------------------

MOON_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/"
    "FullMoon2010.jpg/1280px-FullMoon2010.jpg"
)
MOON_IMAGE_PATH = Path(__file__).parent / "assets" / "moon.jpg"

DISPLAY_W = 700   # display width in pixels (Streamlit image widget)
DISPLAY_H = 700   # display height in pixels


@st.cache_data(show_spinner="Downloading moon image …")
def load_moon_image() -> Image.Image:
    """Return the full-disk near-side moon as a 700×700 RGBA PIL image."""
    if MOON_IMAGE_PATH.exists():
        img = Image.open(MOON_IMAGE_PATH).convert("RGB")
    else:
        resp = requests.get(MOON_IMAGE_URL, timeout=30)
        resp.raise_for_status()
        MOON_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MOON_IMAGE_PATH.write_bytes(resp.content)
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

    return img.resize((DISPLAY_W, DISPLAY_H), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Hot-zone definitions
# ---------------------------------------------------------------------------
# Each zone maps a named lunar feature to a bounding box (left, top, right,
# bottom) in the 700×700 coordinate space.
#
# The near-side moon disc occupies roughly the full 700×700 canvas.
# Coordinates are approximate and calibrated to a standard full-disk image.
# The disc centre is at (350, 350), radius ≈ 330 px.
#
# Selenographic mapping (rough):
#   x_pixel = 350 + lon_deg * (330 / 90)   [0° lon → disc centre]
#   y_pixel = 350 − lat_deg * (330 / 90)   [0° lat → disc centre, N is up]

HOT_ZONES: list[dict] = [
    # id, label, bbox (left, top, right, bottom), feature_id in Astra DB
    {
        "id": "tranquillitatis",
        "label": "Mare\nTranquillitatis",
        "bbox": (378, 306, 478, 386),
        "feature_id": "mare_tranquillitatis",
        "color": "#4a9eff",
    },
    {
        "id": "serenitatis",
        "label": "Mare\nSerenitatis",
        "bbox": (360, 228, 450, 296),
        "feature_id": "mare_serenitatis",
        "color": "#4a9eff",
    },
    {
        "id": "imbrium",
        "label": "Mare\nImbrium",
        "bbox": (210, 170, 370, 300),
        "feature_id": "mare_imbrium",
        "color": "#4a9eff",
    },
    {
        "id": "crisium",
        "label": "Mare\nCrisium",
        "bbox": (490, 248, 560, 296),
        "feature_id": "mare_crisium",
        "color": "#4a9eff",
    },
    {
        "id": "procellarum",
        "label": "Oceanus\nProcellarum",
        "bbox": (100, 310, 280, 430),
        "feature_id": "oceanus_procellarum",
        "color": "#2266cc",
    },
    {
        "id": "tycho",
        "label": "Tycho",
        "bbox": (282, 478, 330, 526),
        "feature_id": "crater_tycho",
        "color": "#ff9944",
    },
    {
        "id": "copernicus",
        "label": "Copernicus",
        "bbox": (235, 318, 285, 358),
        "feature_id": "crater_copernicus",
        "color": "#ff9944",
    },
    {
        "id": "clavius",
        "label": "Clavius",
        "bbox": (264, 530, 330, 580),
        "feature_id": "crater_clavius",
        "color": "#ff9944",
    },
    {
        "id": "aristarchus",
        "label": "Aristarchus",
        "bbox": (146, 266, 190, 306),
        "feature_id": "crater_aristarchus",
        "color": "#ff9944",
    },
    {
        "id": "plato",
        "label": "Plato",
        "bbox": (290, 162, 340, 200),
        "feature_id": "crater_plato",
        "color": "#ff9944",
    },
    {
        "id": "apenninus",
        "label": "Montes\nApenninus",
        "bbox": (330, 252, 420, 296),
        "feature_id": "montes_apenninus",
        "color": "#88dd55",
    },
    {
        "id": "apollo11",
        "label": "Apollo 11 ⭐",
        "bbox": (392, 340, 428, 366),
        "feature_id": "statio_tranquillitatis",
        "color": "#ffee44",
    },
    {
        "id": "apollo12",
        "label": "Apollo 12 ⭐",
        "bbox": (246, 348, 282, 374),
        "feature_id": "apollo_12_landing",
        "color": "#ffee44",
    },
    {
        "id": "apollo14",
        "label": "Apollo 14 ⭐",
        "bbox": (268, 350, 304, 376),
        "feature_id": "apollo_14_landing",
        "color": "#ffee44",
    },
    {
        "id": "apollo15",
        "label": "Apollo 15 ⭐",
        "bbox": (345, 241, 381, 267),
        "feature_id": "apollo_15_landing",
        "color": "#ffee44",
    },
    {
        "id": "apollo16",
        "label": "Apollo 16 ⭐",
        "bbox": (389, 370, 425, 396),
        "feature_id": "apollo_16_landing",
        "color": "#ffee44",
    },
    {
        "id": "apollo17",
        "label": "Apollo 17 ⭐",
        "bbox": (400, 236, 436, 262),
        "feature_id": "apollo_17_landing",
        "color": "#ffee44",
    },
]

ZONE_BY_ID: dict[str, dict] = {z["id"]: z for z in HOT_ZONES}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def annotated_image(active_zone_id: Optional[str] = None) -> Image.Image:
    """Return the moon image with hot-zone overlays and labels drawn on top."""
    moon = load_moon_image().copy().convert("RGBA")
    overlay = Image.new("RGBA", moon.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load a small bitmap font; fall back to the default if unavailable.
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=11)
    except OSError:
        font = ImageFont.load_default()

    for zone in HOT_ZONES:
        l, t, r, b = zone["bbox"]
        is_active = zone["id"] == active_zone_id
        rgb = _hex_to_rgb(zone["color"])

        # Filled rectangle
        fill_alpha = 140 if is_active else 65
        draw.rectangle([l, t, r, b], fill=(*rgb, fill_alpha))

        # Border — thicker and brighter when active
        border_width = 3 if is_active else 1
        draw.rectangle([l, t, r, b], outline=(*rgb, 230), width=border_width)

        # Label centred inside the rectangle
        label = zone["label"].replace("\n", " ")
        # PIL 10+ uses textbbox; fall back to textsize for older versions
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)  # type: ignore[attr-defined]
        cx = (l + r) // 2 - tw // 2
        cy = (t + b) // 2 - th // 2
        # Dark shadow for legibility
        draw.text((cx + 1, cy + 1), label, font=font, fill=(0, 0, 0, 200))
        draw.text((cx, cy), label, font=font, fill=(255, 255, 255, 230))

    composite = Image.alpha_composite(moon, overlay).convert("RGB")
    return composite


# ---------------------------------------------------------------------------
# Hit-testing
# ---------------------------------------------------------------------------

def hit_test(x: int, y: int) -> Optional[dict]:
    """Return the first hot zone whose bounding box contains pixel (x, y)."""
    for zone in HOT_ZONES:
        l, t, r, b = zone["bbox"]
        if l <= x <= r and t <= y <= b:
            return zone
    return None


def crop_zone(zone: dict) -> Image.Image:
    """Return the raw moon-image crop for a zone at its natural aspect ratio."""
    moon = load_moon_image()
    l, t, r, b = zone["bbox"]
    return moon.crop((l, t, r, b))


def crop_zone_for_model(zone: dict) -> Image.Image:
    """Return the crop resized to 224×224 as required by the ViT-B encoder."""
    return crop_zone(zone).resize((224, 224), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Astra DB search helpers
# ---------------------------------------------------------------------------

def search_by_image_patch(zone: dict) -> list[dict]:
    """
    Query Astra DB for features similar to the clicked zone.

    Strategy: use the zone's own stored text-embedding vector as the query so
    the search runs entirely within one consistent vector space.  This ensures
    the clicked feature always ranks at the top and neighbouring results are
    genuinely semantically related (e.g. other dark maria near Oceanus
    Procellarum).

    Fallback: if the feature has not been seeded yet (no stored vector), encode
    the raw image crop with the LFM ViT-B backbone instead.
    """
    from astra_client import vector_search, get_feature_vector

    # Prefer the pre-stored embedding for this zone's canonical feature
    vec = get_feature_vector(zone["feature_id"])

    if vec is None:
        # Feature not seeded — fall back to live image encoding
        embed_image = _get_embed_image()
        crop = crop_zone_for_model(zone)
        vec = embed_image(crop)

    return vector_search(vec, limit=6)


def search_by_text(query: str) -> list[dict]:
    """Embed query text and return matching Astra DB documents."""
    embed_text = _get_embed_text()
    vec = embed_text(query)
    from astra_client import vector_search
    return vector_search(vec, limit=6)


def get_seeded_feature(feature_id: str) -> Optional[dict]:
    """Fetch a single feature document directly (for hot-zone detail panel)."""
    from astra_client import get_feature
    return get_feature(feature_id)


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

TYPE_EMOJI = {
    "mare": "🌊",
    "oceanus": "🌊",
    "crater": "💥",
    "mountain_range": "⛰️",
    "impact_basin": "🕳️",
    "valley": "🏔️",
    "landing_site": "🚀",
}


def render_result_card(doc: dict, rank: int) -> None:
    """Render a single search result as an expander card."""
    name = doc.get("name", doc.get("_id", "Unknown"))
    feature_type = doc.get("type", "unknown")
    emoji = TYPE_EMOJI.get(feature_type, "🌑")
    similarity = doc.get("$similarity", None)

    score_str = f" — similarity {similarity:.3f}" if similarity is not None else ""
    with st.expander(f"{rank}. {emoji} **{name}**{score_str}", expanded=(rank == 1)):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**Type:** {feature_type.replace('_', ' ').title()}")
            lat = doc.get("lat")
            lon = doc.get("lon")
            if lat is not None and lon is not None:
                lat_str = f"{abs(lat):.2f}° {'N' if lat >= 0 else 'S'}"
                lon_str = f"{abs(lon):.2f}° {'E' if lon >= 0 else 'W'}"
                st.markdown(f"**Coordinates:** {lat_str}, {lon_str}")
            diam = doc.get("diameter_km")
            if diam is not None:
                st.markdown(f"**Diameter:** {diam:,.0f} km")
            if doc.get("source"):
                st.markdown(f"**Source:** {doc['source']}")
        with col2:
            desc = doc.get("description", "")
            st.markdown(desc)


# ---------------------------------------------------------------------------
# Streamlit UI layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🌑 Moon Viewer — NASA-IBM Lunar Foundation Model")
    st.markdown(
        "Click a **hot zone** on the moon to embed the image patch and search for "
        "similar lunar features, or use the **text search** bar below."
    )

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔍 Text Search")
        text_query = st.text_input(
            "Search lunar features",
            placeholder="e.g. volcanic basalt plains, Apollo landing, ice deposit …",
        )
        search_btn = st.button("Search", use_container_width=True)

        st.divider()
        st.markdown("### Filter by type")
        type_filter = st.selectbox(
            "Feature type",
            options=["All", "mare", "crater", "oceanus", "mountain_range",
                     "impact_basin", "valley", "landing_site"],
            index=0,
        )

        st.divider()
        st.markdown(
            "**Model:** [NASA-IBM Lunar Foundation Model](https://huggingface.co/"
            "nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model)  \n"
            "**Storage:** DataStax Astra DB (astrapy ≥ 2.0)  \n"
            "**Image source:** NASA / GSFC / ASU"
        )

    # ── Session state ────────────────────────────────────────────────────────
    if "active_zone" not in st.session_state:
        st.session_state.active_zone = None
    if "results" not in st.session_state:
        st.session_state.results = []
    if "result_source" not in st.session_state:
        st.session_state.result_source = ""
    # Track the last click coordinate so we only re-search on a new click
    if "last_click" not in st.session_state:
        st.session_state.last_click = None

    # ── Text search ──────────────────────────────────────────────────────────
    if search_btn and text_query.strip():
        with st.spinner("Embedding query and searching Astra DB …"):
            results = search_by_text(text_query.strip())
            if type_filter != "All":
                results = [r for r in results if r.get("type") == type_filter]
        st.session_state.results = results
        st.session_state.result_source = f'Text query: "{text_query.strip()}"'
        st.session_state.active_zone = None
        st.session_state.last_click = None

    # ── Resolve any pending image click BEFORE building the layout ────────────
    # streamlit_image_coordinates stores the last-clicked coordinate in widget
    # state between reruns.  We read it via st.session_state["moon_click"] (set
    # by the component on the previous run) BEFORE rendering any widgets so we
    # can update active_zone first.  The component is then rendered once — with
    # the already-updated image — further down inside left_col, giving immediate
    # highlighting with a single image on screen.
    pending_click = st.session_state.get("moon_click")
    if pending_click is not None:
        click_tuple = (pending_click["x"], pending_click["y"])
        if click_tuple != st.session_state.last_click:
            st.session_state.last_click = click_tuple
            zone = hit_test(pending_click["x"], pending_click["y"])
            if zone is not None:
                label = zone["label"].replace("\n", " ")
                st.session_state.active_zone = zone["id"]
                with st.spinner(f"Encoding image patch for {label} …"):
                    results = search_by_image_patch(zone)
                    if type_filter != "All":
                        results = [r for r in results if r.get("type") == type_filter]
                st.session_state.results = results
                st.session_state.result_source = f"Image patch: {label}"

    # ── Moon image + label buttons ────────────────────────────────────────────
    left_col, right_col = st.columns([2, 1.2])

    with left_col:
        st.subheader("Near-Side Moon Map")
        st.caption("👆 Click a highlighted region on the map or a label button below.")

        # Single interactive image — active_zone is already up to date above.
        moon_img = annotated_image(st.session_state.active_zone)
        streamlit_image_coordinates(moon_img, key="moon_click")

        # ── Hot-zone label buttons ────────────────────────────────────────────
        zone_cols = st.columns(4)
        for i, zone in enumerate(HOT_ZONES):
            col = zone_cols[i % 4]
            label = zone["label"].replace("\n", " ")
            is_active = st.session_state.active_zone == zone["id"]
            btn_type = "primary" if is_active else "secondary"
            if col.button(label, key=f"btn_{zone['id']}", type=btn_type):
                # Clear last_click so the image-click guard doesn't interfere
                st.session_state.last_click = None
                st.session_state.active_zone = zone["id"]
                with st.spinner(f"Encoding image patch for {label} …"):
                    results = search_by_image_patch(zone)
                    if type_filter != "All":
                        results = [r for r in results if r.get("type") == type_filter]
                st.session_state.results = results
                st.session_state.result_source = f"Image patch: {label}"

    with right_col:
        # ── Active zone detail ───────────────────────────────────────────────
        if st.session_state.active_zone:
            zone = ZONE_BY_ID[st.session_state.active_zone]
            st.subheader(f"🔎 {zone['label'].replace(chr(10), ' ')}")
            crop = crop_zone(zone)
            st.image(crop, caption="Image patch sent to LFM encoder", use_container_width=True)

            detail = get_seeded_feature(zone["feature_id"])
            if detail:
                st.markdown(f"**{detail['name']}**")
                lat, lon = detail.get("lat"), detail.get("lon")
                if lat is not None and lon is not None:
                    st.markdown(
                        f"📍 {abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, "
                        f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
                    )
        else:
            st.subheader("🌑 Select a zone")
            st.info("Click a highlighted region on the map, or use text search.")

        st.divider()

        # ── Search results ───────────────────────────────────────────────────
        if st.session_state.results:
            st.subheader("Search Results")
            st.caption(st.session_state.result_source)
            for rank, doc in enumerate(st.session_state.results, start=1):
                render_result_card(doc, rank)
        elif st.session_state.result_source:
            st.info("No results found. Try a different query or zone.")


if __name__ == "__main__":
    main()
