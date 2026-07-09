"""Room-type taxonomy and parameter distributions for synthetic RIR generation.

DRAFT. Ranges need an architect's eye (Tommi) before wiring the simulation.
simulate.py samples from these and dispatches to a geometry builder per type.

Geometry kinds (pyroomacoustics is not limited to rectangular boxes):
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
frequency-dependent materials. `extra` holds geometry-specific knobs.
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


# DRAFT taxonomy. Validate ranges (and the exotic approximations) before generating.
ROOM_TYPES: list[RoomType] = [
    # --- everyday rooms (the hard discrimination lives here) ---
    # names aligned to the BUT ReverbDB vocabulary (utils.ROOM_LABELS) so real
    # and synthetic manifests share labels directly, no remapping needed.
    RoomType("office",       Geometry.SHOEBOX, (3, 5),   (3, 5),   (2.7, 3.0), (0.20, 0.35), note="furnished, carpet"),
    RoomType("meeting_room", Geometry.SHOEBOX, (5, 8),   (4, 7),   (2.7, 3.2), (0.15, 0.30), note="medium"),
    RoomType("lecture_room", Geometry.SHOEBOX, (8, 20),  (6, 15),  (3.0, 6.0), (0.15, 0.30), note="seating, mixed"),
    RoomType("corridor",     Geometry.SHOEBOX, (15, 40), (1.5, 3), (2.5, 3.5), (0.05, 0.15), note="hard, long and narrow"),
    RoomType("staircase",    Geometry.SHOEBOX, (3, 6),   (3, 6),   (8, 20),    (0.03, 0.10), note="very reflective, tall"),
    RoomType("large_hall",   Geometry.SHOEBOX, (20, 40), (15, 30), (8, 15),    (0.10, 0.25), note="mixed"),
    RoomType("bathroom",     Geometry.SHOEBOX, (2, 4),   (2, 4),   (2.4, 2.8), (0.02, 0.08), note="tiled, very live"),

    # --- exotic / extreme (easy to classify, add range, testable vs OpenAIR reals) ---
    RoomType("cathedral",    Geometry.POLYGON, (25, 60), (12, 25), (12, 30),   (0.04, 0.10),
             extra={"facets": 16, "vaulted": True}, note="stone, huge, very long RT60"),
    RoomType("gas_tank",     Geometry.CYLINDER, (8, 25),  (8, 25),  (10, 30),  (0.01, 0.05),
             extra={"facets": 24}, note="steel cylinder, extreme metallic reverb"),
    RoomType("outdoor_patio", Geometry.PARTIAL, (4, 10),  (3, 8),   (2.5, 4.0), (0.15, 0.30),
             extra={"open_walls": 2, "ground_absorption": (0.2, 0.5)}, note="walls on 1-2 sides + open air"),
    RoomType("outdoor_forest", Geometry.OPEN_FIELD, (30, 80), (30, 80), (15, 30), (0.85, 0.98),
             extra={"ground_absorption": (0.3, 0.7), "scattering": (0.4, 0.8)}, note="near free field + trees scatter"),
]

ROOM_TYPE_NAMES = [rt.name for rt in ROOM_TYPES]
