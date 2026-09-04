"""Room-type taxonomy and parameter distributions for synthetic RIR generation.

simulate.py samples from these and dispatches to a geometry builder per type.

Geometry kinds:
  - shoebox     : rectangular box
  - polygon     : arbitrary floor plan extruded to a height (L-shapes, faceted
                  cathedral). Curves are approximated with flat facets.
  - cylinder    : n-gon prism, approximates round tanks / silos
  - partial     : enclosure with 1-2 open sides (patio); open walls are set to
                  near-total absorption + a reflective ground
  - open_field  : near free field (forest / outdoor). Big absorbing box + a
                  reflective ground + high scattering. An approximation: pra
                  simulates enclosed spaces, true outdoor is out of scope.

Dimensions in meters. Interpretation of (x, y, z) depends on geometry:
  - shoebox / polygon / partial : bounding extents (width, depth, height)
  - cylinder                    : x = diameter, z = height (y ignored)
  - open_field                  : x, y = ground extent, z = height of the "sky" box
`absorption` is a coarse mean absorption range; simulate.py maps it to
frequency-dependent, per-surface materials. `extra` holds geometry-specific
knobs, including `tilt`, the spectral shape of the absorption:
  - "soft" : absorption rises with frequency (carpet, seating, curtains, people;
             offices, meeting/lecture rooms, halls).
  - "hard" : nearly flat, slightly more at low frequency (concrete, tile, stone;
             staircase, corridor, bathroom, cathedral).
  - "neutral": mild rise (default, mixed surfaces).
"""

from dataclasses import dataclass, field
from enum import Enum


class Geometry(str, Enum):
    SHOEBOX = "shoebox"
    POLYGON = "polygon"
    CYLINDER = "cylinder"
    PARTIAL = "partial"
    OPEN_FIELD = "open_field"


@dataclass(frozen=True)
class RoomType:
    name: str
    geometry: Geometry
    x: tuple[float, float]           # see header for interpretation
    y: tuple[float, float]
    z: tuple[float, float]
    absorption: tuple[float, float]  # coarse mean absorption coeff range
    extra: dict = field(default_factory=dict)
    note: str = ""


# Absorption ranges are anchored to published RT60 targets per room use, checked
# with Sabine (RT60 = 0.161*V / (S*mean_alpha)) at the midpoint dimensions. They
# are set from the acoustics literature, NOT fitted to the real BUT medians (the
# real set is the held-out test, tuning to it would leak). Sources:
#   [BB93]     Building Bulletin 93, "Acoustic Design of Schools", UK DfE 2015.
#              Classrooms Tmf <= 0.6s (new build), lecture rooms up to ~1.0s.
#   [S12.60]   ANSI/ASA S12.60-2010: unoccupied core learning space RT <= 0.6-0.7s.
#   [Long]     M. Long, "Architectural Acoustics", 2nd ed., Academic Press 2014.
#              RT tables by use: cellular office ~0.4-0.8s, conference ~0.6-1.0s.
#   [Beranek]  L. Beranek, "Concert Halls and Opera Houses", 2nd ed., Springer
#              2004: mid-freq RT ~1.8-2.1s occupied (multipurpose halls lower).
#   [Kuttruff] H. Kuttruff, "Room Acoustics", 6th ed., CRC Press 2016 (Sabine).
#   [Martellotta] F. Martellotta et al., church/cathedral acoustics: gothic RT ~4-10s.
ROOM_TYPES: list[RoomType] = [
    # --- everyday rooms (the hard discrimination lives here) ---
    # names aligned to the BUT ReverbDB vocabulary (utils.ROOM_LABELS) so real
    # and synthetic manifests share labels directly, no remapping needed.
    RoomType("office",       Geometry.SHOEBOX, (3, 5),   (3, 5),   (2.7, 3.0), (0.12, 0.35), extra={"tilt": "soft"}, note="RT60 ~0.5-1.0s, treated to untreated [Long]"),
    RoomType("meeting_room", Geometry.SHOEBOX, (5, 8),   (4, 7),   (2.7, 3.2), (0.10, 0.28), extra={"tilt": "soft"}, note="RT60 ~0.6-1.0s [Long]"),
    RoomType("lecture_room", Geometry.SHOEBOX, (8, 20),  (6, 15),  (3.0, 6.0), (0.15, 0.30), extra={"tilt": "soft"}, note="RT60 ~0.8-1.0s [BB93, S12.60]"),
    RoomType("corridor",     Geometry.SHOEBOX, (15, 40), (1.5, 3), (2.5, 3.5), (0.05, 0.15), extra={"tilt": "hard"}, note="RT60 ~1.0-2.0s, hard long narrow"),
    RoomType("staircase",    Geometry.SHOEBOX, (3, 6),   (3, 6),   (8, 20),    (0.03, 0.10), extra={"tilt": "hard"}, note="RT60 ~2-4s, concrete, tall"),
    RoomType("large_hall",   Geometry.SHOEBOX, (20, 40), (15, 30), (8, 15),    (0.12, 0.28), extra={"tilt": "soft"}, note="RT60 ~1.5-2.2s, multipurpose [Beranek]"),
    RoomType("bathroom",     Geometry.SHOEBOX, (2, 4),   (2, 4),   (2.4, 2.8), (0.04, 0.12), extra={"tilt": "hard"}, note="RT60 ~0.6-1.2s, tiled live"),

    # --- exotic / extreme: wide acoustic range, no counterpart in the real set ---
    RoomType("cathedral",    Geometry.POLYGON, (25, 60), (12, 25), (12, 30),   (0.04, 0.10),
             extra={"facets": 16, "vaulted": True, "tilt": "hard"}, note="stone, huge, very long RT60"),
    RoomType("outdoor_patio", Geometry.PARTIAL, (4, 10),  (3, 8),   (2.5, 4.0), (0.15, 0.30),
             extra={"open_walls": 2, "ground_absorption": (0.2, 0.5), "tilt": "neutral"}, note="walls on 1-2 sides + open air"),
    RoomType("outdoor_forest", Geometry.OPEN_FIELD, (30, 80), (30, 80), (15, 30), (0.85, 0.98),
             extra={"ground_absorption": (0.3, 0.7), "scattering": (0.4, 0.8), "tilt": "soft"}, note="near free field + trees scatter"),
]

ROOM_TYPE_NAMES = [rt.name for rt in ROOM_TYPES]
