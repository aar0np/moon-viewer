"""
seed_data.py
------------
Populate Astra DB with well-known lunar features and their 768-d embeddings.

Each feature is described as plain text; the text is fed through the
embed_text() function from model.py to produce the embedding stored in the
"$vector" field.  Image-region embeddings are added at run-time via the
interactive app when the user clicks a hot zone.

Usage:
    python seed_data.py

The script is idempotent — re-running it refreshes/replaces the embeddings.
"""

from __future__ import annotations

import os
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
# Authoritative lunar feature catalogue
# ---------------------------------------------------------------------------
#
# Coordinates are selenographic (latitude / longitude in degrees).
# Diameter is in km.  Source: IAU Gazetteer / USGS Planetary Names.
#
FEATURES: list[dict] = [
    # ── MARIA (dark volcanic plains) ──────────────────────────────────────
    {
        "_id": "mare_tranquillitatis",
        "name": "Mare Tranquillitatis",
        "type": "mare",
        "description": (
            "The Sea of Tranquility is a large lunar mare located on the near side "
            "of the Moon. It is composed of dark basaltic rock formed by ancient "
            "volcanic eruptions. Apollo 11, the first crewed lunar landing mission, "
            "touched down on its surface on 20 July 1969. The landing site, known as "
            "Statio Tranquillitatis (Tranquility Base), lies at 0.67°N, 23.47°E."
        ),
        "lat": 8.5,
        "lon": 31.4,
        "diameter_km": 873.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "mare_serenitatis",
        "name": "Mare Serenitatis",
        "type": "mare",
        "description": (
            "The Sea of Serenity is a circular lunar mare northeast of Mare "
            "Tranquillitatis, occupying an ancient impact basin roughly 707 km across. "
            "Its basaltic surface has a slightly higher titanium content than "
            "neighbouring maria, visible as subtle colour differences in multispectral "
            "imagery. Apollo 17, the final crewed lunar landing, landed at Taurus-Littrow "
            "on its southeastern rim in December 1972."
        ),
        "lat": 28.0,
        "lon": 17.5,
        "diameter_km": 707.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "mare_imbrium",
        "name": "Mare Imbrium",
        "type": "mare",
        "description": (
            "The Sea of Rains is the second-largest lunar mare and one of the most "
            "prominent dark regions visible from Earth. It fills the Imbrium impact "
            "basin, which formed about 3.9 billion years ago in the Late Heavy "
            "Bombardment. The basin's circular mountain rings — the Montes Apenninus, "
            "Caucasus, and Alpes — frame the mare like a natural amphitheatre. "
            "Apollo 15 landed near the Apennine foothills in July 1971."
        ),
        "lat": 32.8,
        "lon": -15.6,
        "diameter_km": 1123.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "mare_crisium",
        "name": "Mare Crisium",
        "type": "mare",
        "description": (
            "The Sea of Crises is a notably oval lunar mare on the northeastern limb "
            "of the near side. Its isolation from other maria and its unusually smooth "
            "floor make it one of the most visually distinct features of the lunar disc. "
            "Soviet robotic mission Luna 24 collected regolith samples from Mare Crisium "
            "in 1976, the last robotic sample-return mission from the Moon until China's "
            "Chang'e-5 in 2020."
        ),
        "lat": 17.0,
        "lon": 59.1,
        "diameter_km": 555.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "oceanus_procellarum",
        "name": "Oceanus Procellarum",
        "type": "oceanus",
        "description": (
            "The Ocean of Storms is the only feature on the Moon designated an oceanus. "
            "It is the largest volcanic region on the Moon, covering roughly 2.5 million "
            "square kilometres on the western near side. The region shows evidence of "
            "prolonged and episodic volcanism spanning over a billion years. Apollo 12 "
            "landed in its southeastern portion in November 1969."
        ),
        "lat": -10.0,
        "lon": -57.4,
        "diameter_km": 2568.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "mare_frigoris",
        "name": "Mare Frigoris",
        "type": "mare",
        "description": (
            "The Sea of Cold is a narrow, elongated mare running along the northern "
            "portion of the near side. It extends approximately 1596 km from west to "
            "east and averages only about 100 km in width, making it the most elongated "
            "of the major maria. Its northern boundary approaches 65° latitude and "
            "connects several distinct highland terrains."
        ),
        "lat": 56.0,
        "lon": 1.4,
        "diameter_km": 1596.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "mare_nubium",
        "name": "Mare Nubium",
        "type": "mare",
        "description": (
            "The Sea of Clouds lies in the southern hemisphere of the near side. "
            "It occupies an old, heavily degraded multi-ring impact basin and features "
            "several prominent impact craters on and around its floor, including "
            "Bullialdus. The mare material here is relatively thin, and ancient basin "
            "rings protrude through the basalt as islands of highland terrain."
        ),
        "lat": -21.3,
        "lon": -16.6,
        "diameter_km": 715.0,
        "source": "IAU/USGS",
    },
    # ── PROMINENT CRATERS ─────────────────────────────────────────────────
    {
        "_id": "crater_tycho",
        "name": "Tycho",
        "type": "crater",
        "description": (
            "Tycho is one of the most conspicuous impact craters on the Moon, "
            "famous for its prominent ray system that extends across nearly the entire "
            "near side. It is approximately 86 km in diameter and 4.8 km deep, with a "
            "well-preserved central peak rising about 2 km above its floor. The rays, "
            "composed of bright ejecta, indicate a geologically young age of roughly "
            "108 million years. Surveyor 7 landed near Tycho's northern rim in 1968."
        ),
        "lat": -43.3,
        "lon": -11.2,
        "diameter_km": 86.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "crater_copernicus",
        "name": "Copernicus",
        "type": "crater",
        "description": (
            "Copernicus is a prominent lunar impact crater about 93 km in diameter "
            "located on the Oceanus Procellarum. Often called the 'Monarch of the Moon', "
            "it has terraced inner walls, a complex central peak group rising about "
            "1.2 km, and a well-preserved ray system extending up to 800 km. "
            "The crater is approximately 800 million years old, placing its formation "
            "in the Copernican geologic period named in its honour."
        ),
        "lat": 9.7,
        "lon": -20.0,
        "diameter_km": 93.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "crater_clavius",
        "name": "Clavius",
        "type": "crater",
        "description": (
            "Clavius is one of the largest impact craters on the near side of the Moon, "
            "with a diameter of about 231 km. Its floor hosts a distinctive curved chain "
            "of progressively smaller craters — Porter, Rutherfurd, and others — formed "
            "by secondary impacts. A fictional Moon base named Clavius Base is featured "
            "prominently in Stanley Kubrick's film 2001: A Space Odyssey."
        ),
        "lat": -58.4,
        "lon": -14.4,
        "diameter_km": 231.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "crater_aristarchus",
        "name": "Aristarchus",
        "type": "crater",
        "description": (
            "Aristarchus is the brightest large crater on the near side, with an albedo "
            "significantly higher than its surroundings. It is 40 km in diameter and "
            "3.7 km deep, with complex terraced walls and a central peak. The plateau "
            "to its north, Aristarchus Plateau, is the site of numerous transient lunar "
            "phenomena reports and contains the Schröter's Valley rille — the largest "
            "lunar rille observable from Earth."
        ),
        "lat": 23.7,
        "lon": -47.4,
        "diameter_km": 40.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "crater_plato",
        "name": "Plato",
        "type": "crater",
        "description": (
            "Plato is a prominent walled plain in the northern portion of the lunar "
            "near side, between Mare Imbrium and Mare Frigoris. It is 101 km in "
            "diameter with an unusually flat, dark basaltic floor that was flooded "
            "by lava after the original impact. The dark floor contrasts sharply with "
            "the bright surrounding highlands, making Plato easily recognisable from "
            "Earth with small telescopes."
        ),
        "lat": 51.6,
        "lon": -9.4,
        "diameter_km": 101.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "crater_kepler",
        "name": "Kepler",
        "type": "crater",
        "description": (
            "Kepler is a prominent impact crater on the Oceanus Procellarum, about "
            "32 km in diameter with well-developed ray systems. Like nearby Copernicus, "
            "its bright ejecta rays confirm a geologically recent age. The crater is "
            "named after German astronomer Johannes Kepler, discoverer of the laws of "
            "planetary motion."
        ),
        "lat": 8.1,
        "lon": -38.0,
        "diameter_km": 32.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "surveyor_crater",
        "name": "Surveyor Crater",
        "type": "crater",
        "description": (
            "This crater was named for the Surveyor 3 spacecraft, which landed within "
            "the crater, near the east rim. Apollo 12 landed just north of the Surveyor "
            "Crater on November 19, 1969. Astronauts Pete Conrad and Alan Bean removed "
            "several pieces of Surveyor 3 before returning to the lunar module."
        ),
        "lat": -3.02,
        "lon": -23.42,
        "diameter_km": 0.2,
        "source": "IAU/USGS",
    },
    # ── MOUNTAIN RANGES & HIGHLANDS ───────────────────────────────────────
    {
        "_id": "montes_apenninus",
        "name": "Montes Apenninus",
        "type": "mountain_range",
        "description": (
            "The Lunar Apennines form the southeastern rim of the Mare Imbrium basin, "
            "stretching approximately 600 km. They include Mons Huygens, the highest "
            "point on the Moon's near side at about 5.5 km above the surrounding plains. "
            "Apollo 15 landed at their southern foothills in the Hadley-Apennine region, "
            "where crew explored the Hadley Rille and collected some of the geologically "
            "oldest lunar samples, including the Genesis Rock."
        ),
        "lat": 18.9,
        "lon": -3.1,
        "diameter_km": 600.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "south_pole_aitken_basin",
        "name": "South Pole–Aitken Basin",
        "type": "impact_basin",
        "description": (
            "The South Pole–Aitken Basin is the largest, deepest, and oldest confirmed "
            "impact crater in the Solar System, spanning approximately 2500 km and "
            "reaching depths of up to 8 km below the surrounding surface. Located on "
            "the far side of the Moon, it exposes some of the deepest crustal and "
            "possibly mantle material. Permanently shadowed regions within the basin "
            "are prime candidates for water-ice deposits, making it a priority target "
            "for future exploration by the Artemis programme."
        ),
        "lat": -53.0,
        "lon": 191.0,
        "diameter_km": 2500.0,
        "source": "IAU/USGS",
    },
    {
        "_id": "vallis_schroteri",
        "name": "Vallis Schröteri",
        "type": "valley",
        "description": (
            "Schröter's Valley is the largest known sinuous rille on the Moon, "
            "originating near the Aristarchus Plateau and extending approximately "
            "160 km to the northwest. With a maximum width of about 10 km and depth "
            "reaching 1 km, it is thought to have formed by the collapse of an ancient "
            "lava tube or by the erosive action of flowing lava. It served as a landmark "
            "for early telescopic observers and is named after the German astronomer "
            "Johann Hieronymus Schröter."
        ),
        "lat": 26.2,
        "lon": -50.8,
        "diameter_km": 168.0,
        "source": "IAU/USGS",
    },
    
    {
        "_id": "mount_marilyn",
        "name": "Mount Marilyn",
        "type": "mountain_range",
        "description": (
            "Mount Marilyn (officially Secchi theta) is a lunar mountain within the "
            "Montes Secchi range, situated between Mare Fecunditatis to the east and "
            "Mare Tranquillitatis to the west. It was informally named by astronaut "
            "Jim Lovell for his wife Marilyn during the Apollo 8 mission in 1968, and "
            "the name was officially recognised by the International Astronomical Union "
            "on 26 July 2017. The mountain served as a landmark and navigation reference "
            "during the Apollo missions — it was used by the crew of Apollo 10 to verify "
            "their descent trajectory, and Jim Lovell pointed it out to his wife in the "
            "1995 film Apollo 13. A small crater at its north tip is known as Secchi O."
        ),
        "lat": 1.13,
        "lon": 40.00,
        "diameter_km": None,
        "source": "IAU/USGS",
    },

    # ── APOLLO LANDING SITES ──────────────────────────────────────────────
    {
        "_id": "statio_tranquillitatis",
        "name": "Statio Tranquillitatis (Apollo 11)",
        "type": "landing_site",
        "description": (
            "Tranquility Base is the historic landing site of Apollo 11, the first "
            "crewed Moon landing. Neil Armstrong and Buzz Aldrin touched down on "
            "20 July 1969 at 0°40′27″N, 23°28′23″E in Mare Tranquillitatis. They "
            "spent 2 hours 31 minutes on the surface, collected 21.55 kg of samples, "
            "and left a plaque reading 'We came in peace for all mankind'. The Eagle "
            "Lunar Module descent stage and equipment remain at the site."
        ),
        "lat": 0.674,
        "lon": 23.473,
        "diameter_km": None,
        "source": "NASA",
    },
    {
        "_id": "apollo_12_landing",
        "name": "Apollo 12 Landing Site",
        "type": "landing_site",
        "description": (
            "Apollo 12 landed on 19 November 1969 in Oceanus Procellarum near the "
            "Surveyor 3 spacecraft. Pete Conrad and Alan Bean performed two EVAs, "
            "retrieving parts of Surveyor 3 for analysis and collecting 34.35 kg of "
            "lunar material. The site demonstrated a high-precision landing capability "
            "within walking distance of the unmanned probe."
        ),
        "lat": -3.013,
        "lon": -23.419,
        "diameter_km": None,
        "source": "NASA",
    },
    {
        "_id": "apollo_14_landing",
        "name": "Apollo 14 — Fra Mauro",
        "type": "landing_site",
        "description": (
            "Apollo 14 landed on 5 February 1971 in the Fra Mauro highlands, the "
            "same target region originally assigned to the ill-fated Apollo 13 mission. "
            "Alan Shepard and Edgar Mitchell performed two EVAs, covering 3.45 km on "
            "foot and collecting 42.28 kg of samples including fragments thought to "
            "originate from deep within the lunar crust ejected by the Imbrium impact. "
            "Shepard famously hit two golf balls with a makeshift club during the second "
            "EVA, making him the only person to play golf on the Moon."
        ),
        "lat": -3.645,
        "lon": -17.472,
        "diameter_km": None,
        "source": "NASA",
    },
    {
        "_id": "apollo_15_landing",
        "name": "Apollo 15 — Hadley-Apennine",
        "type": "landing_site",
        "description": (
            "Apollo 15 landed on 30 July 1971 at the base of the Montes Apenninus "
            "beside the Hadley Rille, a 1.5 km wide sinuous channel carved by ancient "
            "lava flows. David Scott and James Irwin were the first to use the Lunar "
            "Roving Vehicle, travelling 27.9 km across the surface over three EVAs. "
            "They collected 77.31 kg of samples including the Genesis Rock — a "
            "4.5-billion-year-old piece of the original lunar crust — from the "
            "slopes of Hadley Delta."
        ),
        "lat": 26.132,
        "lon": 3.634,
        "diameter_km": None,
        "source": "NASA",
    },
    {
        "_id": "apollo_16_landing",
        "name": "Apollo 16 — Descartes Highlands",
        "type": "landing_site",
        "description": (
            "Apollo 16 landed on 21 April 1972 in the Descartes Highlands, the only "
            "crewed mission to explore the lunar central highlands. John Young and "
            "Charles Duke drove the Lunar Roving Vehicle 26.7 km over three EVAs, "
            "collecting 95.71 kg of samples. The mission overturned the pre-flight "
            "hypothesis that the highlands were formed by ancient volcanism; the rocks "
            "proved to be impact breccias, showing the terrain was shaped entirely by "
            "meteorite bombardment."
        ),
        "lat": -8.973,
        "lon": 15.501,
        "diameter_km": None,
        "source": "NASA",
    },
    {
        "_id": "apollo_17_landing",
        "name": "Apollo 17 — Taurus-Littrow Valley",
        "type": "landing_site",
        "description": (
            "Apollo 17 was the final crewed lunar landing, touching down in the "
            "Taurus-Littrow valley on 11 December 1972. Eugene Cernan and geologist "
            "Harrison Schmitt conducted three EVAs totalling over 22 hours, collected "
            "110.52 kg of rock and soil samples, and deployed the Apollo Lunar Surface "
            "Experiments Package. Schmitt discovered orange volcanic glass beads — "
            "evidence of ancient fire-fountain volcanic eruptions."
        ),
        "lat": 20.191,
        "lon": 30.772,
        "diameter_km": None,
        "source": "NASA",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def seed():
    print(f"Embedding {len(FEATURES)} lunar features …")
    docs = []
    for i, feat in enumerate(FEATURES):
        text = f"{feat['name']}: {feat['description']}"
        vec = embed_text(text)
        doc = {**feat, "$vector": vec}
        docs.append(doc)
        print(f"  [{i + 1:2d}/{len(FEATURES)}] {feat['name']}")

    print("\nUpserting to Astra DB …")
    upsert_features(docs)
    total = count_features()
    print(f"Done — collection now contains {total} documents.")


if __name__ == "__main__":
    seed()
