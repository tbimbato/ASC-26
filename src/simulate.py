"""Generate a synthetic, balanced, labelled RIR dataset with pyroomacoustics.

Training data for ASC-26. Real datasets (BUT, MIT survey, OpenAIR) are kept as
a held-out test set (sim-to-real) and handled separately in real_test.py.

STATUS: skeleton. Implement once the taxonomy in room_types.py is validated.
The realism knobs (see simulate_rir) decide the sim-to-real gap.
"""

from pathlib import Path

import numpy as np
# import pyroomacoustics as pra
# import soundfile as sf

from room_types import ROOM_TYPES, Geometry

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sim"
WAV_DIR = OUT_DIR / "wav"
MANIFEST = OUT_DIR / "manifest.csv"

FS = 16000               # match real test sets (BUT is 16 kHz)
N_ROOMS_PER_TYPE = 500   # draft, keep classes balanced
RIRS_PER_ROOM = 1        # source/receiver placements per room
RNG = np.random.default_rng(42)


def sample_room(room_type):
    """Sample concrete dimensions, materials and source/mic positions."""
    raise NotImplementedError  # TODO: draw from room_type ranges + extra


# One builder per geometry kind. Each returns a pyroomacoustics Room.
def build_shoebox(spec):
    raise NotImplementedError  # TODO: pra.ShoeBox

def build_polygon(spec):
    raise NotImplementedError  # TODO: extruded floor plan; faceted for cathedral/curves

def build_cylinder(spec):
    raise NotImplementedError  # TODO: n-gon prism (spec.extra["facets"])

def build_partial(spec):
    raise NotImplementedError  # TODO: box with 1-2 open walls (absorption ~1) + ground

def build_open_field(spec):
    raise NotImplementedError  # TODO: big absorbing box + reflective ground + scattering


BUILDERS = {
    Geometry.SHOEBOX: build_shoebox,
    Geometry.POLYGON: build_polygon,
    Geometry.CYLINDER: build_cylinder,
    Geometry.PARTIAL: build_partial,
    Geometry.OPEN_FIELD: build_open_field,
}


def simulate_rir(spec):
    """Build the room (dispatch by geometry) and return one RIR at FS.

    Realism knobs (tune to close the sim-to-real gap):
      - image-source + ray tracing (scattering)
      - frequency-dependent absorption (materials), not a single coefficient
      - source/mic positions kept away from walls
      - additive noise / finite SNR, band-limit to FS
    """
    room = BUILDERS[spec.geometry](spec)
    raise NotImplementedError  # TODO: place src/mic, compute_rir, return ir


def main():
    """Loop types -> sample rooms -> simulate RIRs -> save wavs + manifest.csv.

    Manifest columns: room_id, label, geometry, path
    (label = room-type name; one row per generated RIR).
    """
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError  # TODO


if __name__ == "__main__":
    main()
