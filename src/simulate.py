"""Generate a synthetic, balanced, labelled RIR dataset with pyroomacoustics.

Training data for ASC-26. The real corpora (BUT, AIR, ACE) are the held-out test
set and are assembled separately in real_test.py.

One builder per geometry kind (shoebox, polygon, cylinder, partial, open_field).
Absorption is frequency-dependent, a coefficient per octave band shaped by the
room type's `tilt`, and for shoebox rooms drawn per surface around the type's
mean so no box is perfectly uniform. Scattering is kept for the diffuse and
exotic cases.

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

from room_types import ROOM_TYPES, ROOM_TYPE_NAMES, Geometry

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sim"
WAV_DIR = OUT_DIR / "wav"
MANIFEST = OUT_DIR / "manifest.csv"

FS = 16000        # match real test sets (BUT is 16 kHz)
MAX_ORDER = 3     # image-source order; ray tracing carries the late tail
MIN_SRC_MIC_DIST = 1.0
MARGIN = 0.6      # keep sources/mics away from surfaces (m)
# Ray-tracing receiver sphere. The automatic ray count scales ~1/radius^2, so a
# larger sphere is much cheaper on the big rooms. At 0.8 m the six features move
# less than the room-to-room variance, and the ~2.3 ms of arrival smearing stays
# under the 2.5 ms DRR direct window.
RECEIVER_RADIUS = 0.8

# Octave band centers for frequency-dependent materials (up to fs/2 = 8 kHz).
OCTAVE_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]

# Spectral shape of the absorption per tilt, one multiplier per octave band,
# roughly mean-1 so it redistributes a given mean absorption across frequency
# without changing its overall level much. See room_types.py for the physics.
TILT_SHAPES = {
    "soft":    [0.55, 0.70, 0.90, 1.05, 1.25, 1.45, 1.55],  # rises with freq
    "hard":    [1.25, 1.15, 1.05, 1.00, 0.92, 0.85, 0.80],  # flat, slight low
    "neutral": [0.80, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20],  # mild rise
}


# --- helpers ---------------------------------------------------------------

def _u(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _mat(absorption, scattering=None):
    return pra.Material(absorption, scattering) if scattering is not None else pra.Material(absorption)


def _freq_mat(rng, mean_abs, tilt, scattering=None, surface_factor=1.0):
    """A frequency-dependent pra.Material: the mean absorption is spread over
    the octave bands by the tilt shape, jittered a little per band, and scaled
    by an optional per-surface factor. Coefficients are clipped to a physical
    (0.01, 0.99) range."""
    shape = TILT_SHAPES.get(tilt, TILT_SHAPES["neutral"])
    coeffs = []
    for m in shape:
        a = mean_abs * m * surface_factor * rng.uniform(0.9, 1.1)
        coeffs.append(float(np.clip(a, 0.01, 0.99)))
    ea = {"coeffs": coeffs, "center_freqs": OCTAVE_BANDS}
    return pra.Material(ea, scattering) if scattering is not None else pra.Material(ea)


def _shoebox_materials(rng, spec):
    """Per-surface frequency-dependent materials for a shoebox. Each of the six
    surfaces gets its own absorption factor around the room mean, so the box is
    not perfectly uniform (real rooms never are: carpet floor, tiled ceiling,
    mixed walls)."""
    tilt = spec["extra"].get("tilt", "neutral")
    a, s = spec["a"], spec["scat"]
    surfaces = ["east", "west", "north", "south", "ceiling", "floor"]
    return {w: _freq_mat(rng, a, tilt, s, surface_factor=rng.uniform(0.8, 1.25))
            for w in surfaces}


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

def build_shoebox(spec, mats=None, rng=None):
    Lx, Ly, Lz = spec["Lx"], spec["Ly"], spec["Lz"]
    if mats is None:
        mats = _shoebox_materials(rng, spec)
    room = pra.ShoeBox([Lx, Ly, Lz], materials=mats, **_new_room())
    return room, lambda rng: _box_points(rng, Lx, Ly, Lz)


def build_partial(spec, rng):
    e, a, s, tilt = spec["extra"], spec["a"], spec["scat"], spec["extra"].get("tilt", "neutral")
    ground = _u(rng, e.get("ground_absorption", (0.2, 0.5)))
    walls = ["north", "south", "east", "west"]
    k = min(int(e.get("open_walls", 2)), 4)
    openw = set(rng.choice(walls, size=k, replace=False).tolist())
    mats = {w: (_mat(0.99) if w in openw else _freq_mat(rng, a, tilt, s)) for w in walls}
    mats["ceiling"] = _mat(0.99)                    # open sky
    mats["floor"] = _freq_mat(rng, ground, "hard")  # ground
    return build_shoebox(spec, mats)


def build_open_field(spec, rng):
    e = spec["extra"]
    ground = _u(rng, e.get("ground_absorption", (0.3, 0.7)))
    scat = _u(rng, e.get("scattering", (0.4, 0.8)))
    sky = _mat(0.98, scat)
    mats = {w: sky for w in ["north", "south", "east", "west"]}
    mats["ceiling"] = _mat(0.98, scat)
    mats["floor"] = _freq_mat(rng, ground, "soft", scat)  # foliage/ground damps highs
    return build_shoebox(spec, mats)


def build_polygon(spec, rng):
    n = int(spec["extra"].get("facets", 12))
    rx, ry = spec["Lx"] / 2, spec["Ly"] / 2
    corners = _ngon(rx, ry, n, rx, ry)
    tilt = spec["extra"].get("tilt", "hard")
    mats = _freq_mat(rng, spec["a"], tilt, spec["scat"])
    room = pra.Room.from_corners(corners, materials=mats, **_new_room())
    room.extrude(spec["Lz"], materials=mats)
    R = 0.5 * min(rx, ry)
    return room, lambda rng: _disk_points(rng, rx, ry, R, spec["Lz"])


def build_cylinder(spec, rng):
    n = int(spec["extra"].get("facets", 24))
    R = spec["Lx"] / 2
    corners = _ngon(R, R, n, R, R)
    tilt = spec["extra"].get("tilt", "hard")
    mats = _freq_mat(rng, spec["a"], tilt, spec["scat"])
    room = pra.Room.from_corners(corners, materials=mats, **_new_room())
    room.extrude(spec["Lz"], materials=mats)
    return room, lambda rng: _disk_points(rng, R, R, 0.5 * R, spec["Lz"])


def build_room(spec, rng):
    g = spec["geometry"]
    if g == Geometry.SHOEBOX:
        return build_shoebox(spec, rng=rng)
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
    room.set_ray_tracing(receiver_radius=RECEIVER_RADIUS)
    src, mic = placer(rng)
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    ir = np.asarray(room.rir[0][0], dtype=np.float32)
    peak = float(np.max(np.abs(ir))) or 1.0
    return ir / peak


def _status(done, total, start):
    el = time.time() - start
    eta = el / done * (total - done) if done else 0
    sys.stdout.write(f"\r  [{done:>5}/{total}] {done / total * 100:5.1f}%  "
                     f"{el:5.0f}s elapsed  ETA {eta:5.0f}s   ")
    sys.stdout.flush()


def _simulate_room(task):
    """Worker: simulate one room (all its RIRs) and write the wavs. Rooms are
    independent, so this is called in parallel across processes. Each room gets
    its own RNG from an independent child seed, so parallelism does not change
    (or correlate) the sampled rooms. Returns the manifest rows for this room."""
    rt_idx, room_i, child_seed, rirs_per_room = task
    rt = ROOM_TYPES[rt_idx]
    rng = np.random.default_rng(child_seed)
    spec = sample_room(rt, rng)
    room_id = f"{rt.name}_{room_i:04d}"
    rows = []
    for j in range(rirs_per_room):
        try:
            ir = simulate_rir(spec, rng)
        except Exception as e:
            print(f"\n  SKIP {room_id} r{j}: {e}")
            continue
        fn = f"{room_id}_r{j}.wav"
        sf.write(WAV_DIR / fn, ir, FS)
        rows.append({"room_id": room_id, "label": rt.name,
                     "geometry": rt.geometry.value,
                     "path": str((WAV_DIR / fn).relative_to(ROOT))})
    return rows


def generate(n_per_type, rirs_per_room, seed, workers):
    from multiprocessing import Pool

    WAV_DIR.mkdir(parents=True, exist_ok=True)

    # one task per room, each with an independent child seed
    seeds = np.random.SeedSequence(seed).spawn(n_per_type * len(ROOM_TYPES))
    tasks, k = [], 0
    for rt_idx in range(len(ROOM_TYPES)):
        for i in range(n_per_type):
            tasks.append((rt_idx, i, seeds[k], rirs_per_room))
            k += 1

    total = len(tasks)
    start = time.time()
    all_rows, done = [], 0
    print(f"generating {total} rooms x {rirs_per_room} RIR on {workers} workers "
          f"(receiver_radius={RECEIVER_RADIUS})")

    if workers == 1:  # serial path, for debugging and --smoke
        for t in tasks:
            all_rows.extend(_simulate_room(t))
            done += 1
            _status(done, total, start)
    else:
        with Pool(processes=workers) as pool:
            for rows in pool.imap_unordered(_simulate_room, tasks, chunksize=4):
                all_rows.extend(rows)
                done += 1
                _status(done, total, start)
    sys.stdout.write("\n")

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["room_id", "label", "geometry", "path"])
        w.writeheader()
        w.writerows(all_rows)

    counts = {}
    for r in all_rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print("\nper type:")
    for name in ROOM_TYPE_NAMES:
        print(f"  {name:16s} {counts.get(name, 0)}")
    skipped = total * rirs_per_room - len(all_rows)
    print(f"\nwrote {len(all_rows)} RIRs ({skipped} skipped) in "
          f"{time.time() - start:.0f}s -> {MANIFEST}")


def main():
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-type", type=int, default=200)
    p.add_argument("--rirs-per-room", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="parallel processes (default: cores-1); 1 = serial")
    p.add_argument("--smoke", action="store_true", help="tiny run (5 per type)")
    a = p.parse_args()
    n = 5 if a.smoke else a.n_per_type
    generate(n, a.rirs_per_room, a.seed, a.workers)


if __name__ == "__main__":
    main()
