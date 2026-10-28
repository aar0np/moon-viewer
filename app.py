"""
app_csv.py
----------
Moon Viewer — interactive Streamlit application (CSV-driven hot zones).

Identical to app.py except that HOT_ZONES is built dynamically from
data/moon_poi.csv rather than being hard-coded.  Bounding boxes are
computed from each feature's selenographic coordinates using an
orthographic projection onto the disc in the source image.  The photo
shows libration and is slightly rotated, so the projection is centred on
a fitted sub-observer point (_SUB_LAT, _SUB_LON) and rotated by _ROT_DEG;
the result is then scaled to the 700×700 display canvas.

Features on the far side (not visible from the sub-observer point) are
skipped automatically, e.g. the South Pole–Aitken Basin.

Box half-size scales with diameter_km (clamped to a minimum of 18 px so
small craters and landing sites remain clickable).

Run:
    streamlit run app_csv.py
"""

from __future__ import annotations

import csv
import io
import math
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
    page_title="Moon POI Viewer — NASA-IBM LFM",
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
# Hot-zone construction from CSV
# ---------------------------------------------------------------------------

# Disc geometry in source-image pixels (measured on assets/moon.jpg, 1280×1216)
_SRC_W      = 1280
_SRC_H      = 1216
_DISC_CX    = 640     # pixel x of disc centre
_DISC_CY    = 620     # pixel y of disc centre
_DISC_R     = 532     # disc radius in pixels

# Viewing geometry, least-squares fitted to Tycho, Plato, Aristarchus, Kepler
# and Grimaldi (residuals < 4 px)
_SUB_LAT    = -5.73   # selenographic latitude facing the camera (°)
_SUB_LON    = -3.60   # selenographic longitude facing the camera (°)
_ROT_DEG    = -15.25  # rotation of lunar north from image up (°)

# Source → display scale factors (the display canvas is stretched to 700×700)
_SX = DISPLAY_W / _SRC_W
_SY = DISPLAY_H / _SRC_H

# Per-type colour palette
TYPE_COLOR: dict[str, str] = {
    "mare":           "#4a9eff",
    "oceanus":        "#2266cc",
    "crater":         "#ff9944",
    "mountain_range": "#88dd55",
    "impact_basin":   "#88dd55",
    "valley":         "#88dd55",
    "landing_site":   "#ffee44",
}
_DEFAULT_COLOR = "#aaaaaa"

# Box half-size scaling: larger features get bigger boxes, with min/max clamps.
_MIN_HALF = 18    # px — keeps tiny features (landing sites) clickable
_MAX_HALF = 90    # px — prevents huge maria from dominating the whole disc
_DIAM_SCALE = 0.06  # px per km of diameter


def _half_size(diameter_km: Optional[float]) -> int:
    """Return the box half-size in pixels for a feature of given diameter."""
    if diameter_km is None or diameter_km == 0:
        return _MIN_HALF
    return int(max(_MIN_HALF, min(_MAX_HALF, diameter_km * _DIAM_SCALE)))


def _label(name: str, feature_type: str) -> str:
    """Build a display label for a feature.

    Landing sites are condensed to 'Apollo NN ⭐' by extracting the mission
    number from anywhere in the name (handles both 'Apollo 12 Landing Site'
    and 'Statio Tranquillitatis (Apollo 11)').

    Other multi-word names are split across two lines so the label fits inside
    its bounding box (mirrors the hand-authored style in app.py).
    """
    if feature_type == "landing_site":
        import re
        match = re.search(r"Apollo\s+(\d+)", name, re.IGNORECASE)
        if match:
            return f"Apollo {int(match.group(1)):02d} ⭐"
        return name + " ⭐"

    # Non-landing-site: split long names onto two lines
    words = name.split()
    if len(words) >= 3:
        mid = len(words) // 2
        return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    return name


CSV_PATH = Path(__file__).parent / "data" / "moon_poi.csv"


@st.cache_data(show_spinner=False)
def build_hot_zones() -> list[dict]:
    """
    Read moon_poi.csv and compute a HOT_ZONES list.

    Each row becomes one zone dict with keys:
        id, label, bbox (left, top, right, bottom), feature_id, color

    Rows whose projected centre lies outside the near-side disc are silently
    skipped (e.g. far-side features with lon > 90° or lon < −90°).

    Zones are sorted largest-first so that hit_test() can find small features
    that overlap larger ones by iterating in reverse.
    """
    zones: list[dict] = []

    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lat = float(row["lat"])
            lon = float(row["lon"])
            raw_diam = row["diameter_km"].strip()
            diameter_km = float(raw_diam) if raw_diam else None
            feature_type = row["type"].strip()
            fid = row["_id"].strip()

            # Orthographic projection centred on the sub-observer point
            lat_r = math.radians(lat)
            dlon = math.radians(lon - _SUB_LON)
            b0 = math.radians(_SUB_LAT)
            xp = math.cos(lat_r) * math.sin(dlon)
            yp = math.cos(b0) * math.sin(lat_r) - math.sin(b0) * math.cos(lat_r) * math.cos(dlon)

            # Skip far-side features
            if math.sin(b0) * math.sin(lat_r) + math.cos(b0) * math.cos(lat_r) * math.cos(dlon) < 0:
                continue

            # Rotate by the image's position angle, then scale to display
            rot = math.radians(_ROT_DEG)
            x = xp * math.cos(rot) - yp * math.sin(rot)
            y = xp * math.sin(rot) + yp * math.cos(rot)
            cx = (_DISC_CX + _DISC_R * x) * _SX
            cy = (_DISC_CY - _DISC_R * y) * _SY

            half = _half_size(diameter_km)
            bbox = (
                int(cx - half),
                int(cy - half),
                int(cx + half),
                int(cy + half),
            )

            zones.append({
                "id":         fid,
                "label":      _label(row["name"].strip(), feature_type),
                "bbox":       bbox,
                "feature_id": fid,
                "color":      TYPE_COLOR.get(feature_type, _DEFAULT_COLOR),
            })

    # Sort largest bounding-box area first; hit_test reverses the list so
    # smaller zones on top are matched before the larger ones beneath them.
    zones.sort(key=lambda z: (z["bbox"][2] - z["bbox"][0]) * (z["bbox"][3] - z["bbox"][1]), reverse=True)
    return zones


HOT_ZONES: list[dict] = build_hot_zones()
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

        # Strip the star emoji for the overlay label
        index = label.find(" ⭐")
        if index > -1:
            label = label[:index]

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
    # Iterate in reverse so smaller zones (appended later after sort reversal)
    # are tested before the large zones that contain them.
    for zone in reversed(HOT_ZONES):
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
    st.title("Moon POI Viewer — NASA-IBM Lunar Foundation Model")
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
        search_btn = st.button("Search", width='content')

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
            "**Storage:** DataStax Astra DB  \n"
            "**Image source:** NASA / GSFC / ASU"
        )

    # ── Session state ────────────────────────────────────────────────────────
    if "active_zone" not in st.session_state:
        st.session_state.active_zone = None
    if "results" not in st.session_state:
        st.session_state.results = []
    if "result_source" not in st.session_state:
        st.session_state.result_source = ""
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
        st.caption("Click a highlighted region on the map or a label button below.")

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
            st.image(crop, caption="Image patch sent to LFM encoder", width='content')

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
            st.subheader("Select a zone")
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
