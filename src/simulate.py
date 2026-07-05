"""Generate a synthetic, balanced, labelled RIR dataset with pyroomacoustics.

Training data for ASC-26. Real datasets (BUT, MIT survey, OpenAIR) are kept as
a held-out test set (sim-to-real) and handled separately in real_test.py.
(real ds are not downloaded yet at this stage!)

One builder per geometry kind (shoebox, polygon, cylinder, partial, open_field).
v1 uses a single (frequency-independent) absorption per room drawn from the type
range, plus scattering for the diffuse/exotic cases. Frequency-dependent
materials are a later refinement.

    python src/simulate.py --smoke            # tiny run to sanity-check
    python src/simulate.py --n-per-type 300   # full run
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf

from room_types import ROOM_TYPES, Geometry

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sim"
WAV_DIR = OUT_DIR / "wav"
MANIFEST = OUT_DIR / "manifest.csv"

FS = 16000        # match real test sets (BUT is 16 kHz)
MAX_ORDER = 3     # image-source order; ray tracing carries the late tail
MIN_SRC_MIC_DIST = 1.0
MARGIN = 0.6      # keep sources/mics away from surfaces (m)


# --- helpers ---------------------------------------------------------------

def _u(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _mat(absorption, scattering=None):
    return pra.Material(absorption, scattering) if scattering is not None else pra.Material(absorption)


def _ngon(rx, ry, n, cx, cy):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.vstack([cx + rx * np.cos(ang), cy + ry * np.sin(ang)])


def _new_room(**kw):
    return dict(fs=FS, max_order=MAX_ORDER, ray_tracing=True, air_absorption=True, **kw)


def _box_points(rng, Lx, Ly, Lz):
    def pt():
        return np.array([rng.uniform(MARGIN, Lx - MARGIN),
                         rng.uniform(MARGIN, Ly - MARGIN),
                         rng.uniform(MARGIN, Lz - MARGIN)])
    s, m = pt(), pt()
    for _ in range(50):
        if np.linalg.norm(s - m) >= MIN_SRC_MIC_DIST:
            break
        m = pt()
    return s, m


def _disk_points(rng, cx, cy, R, Lz):
    def pt():
        a = rng.uniform(0, 2 * np.pi)
        r = R * np.sqrt(rng.uniform(0, 1))
        return np.array([cx + r * np.cos(a), cy + r * np.sin(a),
                         rng.uniform(MARGIN, Lz - MARGIN)])
    s, m = pt(), pt()
    for _ in range(50):
        if np.linalg.norm(s - m) >= MIN_SRC_MIC_DIST:
            break
        m = pt()
    return s, m


# --- sampling --------------------------------------------------------------

def sample_room(rt, rng):
    return {
        "name": rt.name,
        "geometry": rt.geometry,
        "Lx": _u(rng, rt.x), "Ly": _u(rng, rt.y), "Lz": _u(rng, rt.z),
        "a": _u(rng, rt.absorption),
        "scat": _u(rng, rt.extra["scattering"]) if "scattering" in rt.extra else 0.1,
        "extra": rt.extra,
    }


# --- geometry builders -----------------------------------------------------
# Each returns (room, placer) where placer(rng) -> (src_xyz, mic_xyz).

def build_shoebox(spec, mats=None):
    Lx, Ly, Lz = spec["Lx"], spec["Ly"], spec["Lz"]
    mats = mats if mats is not None else _mat(spec["a"], spec["scat"])
    room = pra.ShoeBox([Lx, Ly, Lz], materials=mats, **_new_room())
    return room, lambda rng: _box_points(rng, Lx, Ly, Lz)


def build_partial(spec, rng):
    e, a, s = spec["extra"], spec["a"], spec["scat"]
    ground = _u(rng, e.get("ground_absorption", (0.2, 0.5)))
    walls = ["north", "south", "east", "west"]
    k = min(int(e.get("open_walls", 2)), 4)
    openw = set(rng.choice(walls, size=k, replace=False).tolist())
    mats = {w: (_mat(0.99) if w in openw else _mat(a, s)) for w in walls}
    mats["ceiling"] = _mat(0.99)      # open sky
    mats["floor"] = _mat(ground)      # ground
    return build_shoebox(spec, mats)


def build_open_field(spec, rng):
    e = spec["extra"]
    ground = _u(rng, e.get("ground_absorption", (0.3, 0.7)))
    scat = _u(rng, e.get("scattering", (0.4, 0.8)))
    sky = _mat(0.98, scat)
    mats = {w: sky for w in ["north", "south", "east", "west"]}
    mats["ceiling"] = _mat(0.98, scat)
    mats["floor"] = _mat(ground, scat)
    return build_shoebox(spec, mats)


def build_polygon(spec, rng):
    n = int(spec["extra"].get("facets", 12))
    rx, ry = spec["Lx"] / 2, spec["Ly"] / 2
    corners = _ngon(rx, ry, n, rx, ry)
    mats = _mat(spec["a"], spec["scat"])
    room = pra.Room.from_corners(corners, materials=mats, **_new_room())
    room.extrude(spec["Lz"], materials=mats)
    R = 0.5 * min(rx, ry)
    return room, lambda rng: _disk_points(rng, rx, ry, R, spec["Lz"])


def build_cylinder(spec, rng):
    n = int(spec["extra"].get("facets", 24))
    R = spec["Lx"] / 2
    corners = _ngon(R, R, n, R, R)
    mats = _mat(spec["a"], spec["scat"])
    room = pra.Room.from_corners(corners, materials=mats, **_new_room())
    room.extrude(spec["Lz"], materials=mats)
    return room, lambda rng: _disk_points(rng, R, R, 0.5 * R, spec["Lz"])


def build_room(spec, rng):
    g = spec["geometry"]
    if g == Geometry.SHOEBOX:
        return build_shoebox(spec)
    if g == Geometry.PARTIAL:
        return build_partial(spec, rng)
    if g == Geometry.OPEN_FIELD:
        return build_open_field(spec, rng)
    if g == Geometry.POLYGON:
        return build_polygon(spec, rng)
    if g == Geometry.CYLINDER:
        return build_cylinder(spec, rng)
    raise ValueError(g)


# --- simulation ------------------------------------------------------------

def simulate_rir(spec, rng):
    room, placer = build_room(spec, rng)
    src, mic = placer(rng)
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    ir = np.asarray(room.rir[0][0], dtype=np.float32)
    peak = float(np.max(np.abs(ir))) or 1.0
    return ir / peak


def _status(done, total, msg, start, skipped):
    el = time.time() - start
    eta = el / done * (total - done) if done else 0
    sys.stdout.write(f"\r  [{done:>5}/{total}] {done / total * 100:5.1f}%  "
                     f"{msg:<15} {skipped} skip  {el:4.0f}s elapsed  ETA {eta:4.0f}s   ")
    sys.stdout.flush()


def generate(n_per_type, rirs_per_room, seed):
    rng = np.random.default_rng(seed)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    total = n_per_type * len(ROOM_TYPES) * rirs_per_room
    start = time.time()
    done = skipped = 0
    rows = []
    counts = {}

    for rt in ROOM_TYPES:
        made = 0
        for i in range(n_per_type):
            spec = sample_room(rt, rng)
            room_id = f"{rt.name}_{i:04d}"
            for j in range(rirs_per_room):
                try:
                    ir = simulate_rir(spec, rng)
                except Exception as e:
                    skipped += 1
                    sys.stdout.write("\n")
                    print(f"  SKIP {room_id} r{j}: {e}")
                    continue
                fn = f"{room_id}_r{j}.wav"
                sf.write(WAV_DIR / fn, ir, FS)
                rows.append({"room_id": room_id, "label": rt.name,
                             "geometry": rt.geometry.value,
                             "path": str((WAV_DIR / fn).relative_to(ROOT))})
                made += 1
                done += 1
                _status(done, total, rt.name, start, skipped)
        counts[rt.name] = made
    sys.stdout.write("\n")

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["room_id", "label", "geometry", "path"])
        w.writeheader()
        w.writerows(rows)

    print("\nper type:")
    for name, c in counts.items():
        print(f"  {name:16s} {c}")
    print(f"\nwrote {len(rows)} RIRs ({skipped} skipped) in {time.time() - start:.0f}s -> {MANIFEST}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-type", type=int, default=200)
    p.add_argument("--rirs-per-room", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="tiny run (5 per type)")
    a = p.parse_args()
    n = 5 if a.smoke else a.n_per_type
    generate(n, a.rirs_per_room, a.seed)


if __name__ == "__main__":
    main()
