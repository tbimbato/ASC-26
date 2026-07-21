"""Assemble the real held-out test set for sim-to-real evaluation.

Combines real RIR datasets into one manifest with the same room-type vocabulary
as the synthetic set (room_types.py / utils.ROOM_LABELS), so build_dataset.py
and train.py treat it exactly like any other manifest.

  - BUT ReverbDB : few rooms, thousands of measurements each. Subsampled to
                   `per_room` RIRs so no single room dominates.
  - AIR (Aachen) : binaural (full-band) recordings only. A handful of spaces,
                   several RIRs each (like BUT). Room id is the space name; the
                   _mls duplicate of each _logsweep is dropped.
  - ACE Challenge: 7 real rooms (2x office, meeting_room, lecture_room + 1
                   building lobby, unmapped/skipped), 2 mic/source positions
                   each = 2 RIRs per room. Single-channel mic only (the corpus
                   also ships EM32/crucifix/mobile arrays and a Chromebook
                   capture; those are multichannel or bandlimited, out of
                   scope). The *_Noise_* files in each position folder are
                   long babble/fan/ambient recordings for the denoising
                   challenge, not RIRs; only *_RIR.wav is used. Small by
                   design (it's a characterisation benchmark, not a bulk
                   corpus) but each room ships literature-grade ground-truth
                   RT60/DRR, useful to sanity-check extract_features.

Deliberately excluded, to keep the physical decay features (RT60, C80, DRR)
honest:
  - MIT survey   : most IRs have too little dynamic range before the ambient
                   noise floor (median usable decay ~0.7s) to support a T30-style
                   RT60 estimate. Perfectly valid for its own purpose (perceptual
                   statistics of everyday reverb, Traer & McDermott PNAS 2016),
                   but not for reliable RT60/C80/DRR estimation. Unusable here.
  - AIR phone    : telephone-band (300-3400 Hz) recordings. The band limit
                   inflates C80 (removes low-freq reverb energy): a capture-bias
                   confound we keep out of the main test set.

The "room" is the grouping unit for honest evaluation: BUT, AIR and ACE
contribute few rooms with several RIRs (capped at `per_room`; ACE positions
1/2 of the same room are pooled as one room, not two). Everything not covered
by the taxonomy is skipped.

    python src/real_test.py                  # default: 5 RIRs/room
    python src/real_test.py --per-room 10
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

from utils import ROOM_LABELS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "real"
MANIFEST = OUT_DIR / "manifest.csv"

BUT_DIR = RAW / "BUT_ReverbDB_rel_19_06_RIR-Only"
BUT_TARGET = "IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav"
AIR_WAV_DIR = RAW / "AIR_1_4" / "AIR_wav_files"
# folder name is the dataset's own long Zenodo title; matched by prefix
ACE_DIR = next((d for d in RAW.glob("ACE*") if d.is_dir()), None)

# AIR: match on the space token (alpha part) after stripping air_/phone_ prefix.
AIR_MAP = {
    "office": "office",
    "meeting": "meeting_room",
    "lecture": "lecture_room", "lecutre": "lecture_room",  # source typo
    "stairway": "staircase",
    "bathroom": "bathroom",
}

# ACE room dir names (Office_1, Meeting_Room_2, ...) minus the trailing index.
ACE_MAP = {
    "office": "office",
    "meeting_room": "meeting_room",
    "lecture_room": "lecture_room",
}


def collect_but() -> dict[str, list[Path]]:
    by_room: dict[str, list[Path]] = {r: [] for r in ROOM_LABELS}
    for wav in BUT_DIR.rglob(BUT_TARGET):
        if wav.parent.name != "RIR":
            continue
        room_id = wav.relative_to(BUT_DIR).parts[0]
        if room_id in by_room:
            by_room[room_id].append(wav)
    return {f"but_{k}": v for k, v in by_room.items() if v}


def collect_air() -> dict[str, list[Path]]:
    """Binaural only; group by space token; keep one of each _mls/_logsweep pair."""
    by_room: dict[str, list[Path]] = {}
    for wav in sorted(AIR_WAV_DIR.glob("*.wav")):
        if "phone" in wav.stem or "_mls" in wav.stem:  # bandlimited / duplicate
            continue
        token = wav.stem
        token = token[len("air_"):] if token.startswith("air_") else token
        m = re.match(r"([a-z]+)([0-9]*)", token)  # space name + optional index
        if not m or m.group(1) not in AIR_MAP:
            continue
        space = m.group(1) + m.group(2)
        by_room.setdefault(f"air_{space}", []).append(wav)
    return by_room


def collect_ace() -> dict[str, list[Path]]:
    """Pool both mic/source positions of each room as one room (not two)."""
    if ACE_DIR is None:
        return {}
    single_dir = ACE_DIR / "Single"
    by_room: dict[str, list[Path]] = {}
    for room_dir in sorted(single_dir.iterdir()):
        if not room_dir.is_dir():
            continue
        m = re.match(r"(.+)_(\d+)$", room_dir.name)  # e.g. Office_1 -> Office
        if not m or m.group(1).lower() not in ACE_MAP:
            continue  # Building_Lobby (no trailing index) doesn't map, skipped
        rirs = sorted(room_dir.glob("*/*_RIR.wav"))
        if rirs:
            by_room[f"ace_{room_dir.name.lower()}"] = rirs
    return by_room


def label_for(room_id: str) -> str:
    if room_id.startswith("but_"):
        return ROOM_LABELS[room_id[len("but_"):]]
    if room_id.startswith("air_"):
        return AIR_MAP[re.match(r"air_([a-z]+)", room_id).group(1)]
    if room_id.startswith("ace_"):
        base = re.sub(r"_\d+$", "", room_id[len("ace_"):])
        return ACE_MAP[base]
    raise ValueError(f"unknown source for {room_id}")


def build(per_room: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rooms: dict[str, list[Path]] = {}
    rooms.update(collect_but())
    rooms.update(collect_air())
    rooms.update(collect_ace())

    rows = []
    for room_id, paths in rooms.items():
        label = label_for(room_id)
        k = min(per_room, len(paths))
        chosen = rng.choice(paths, size=k, replace=False) if k < len(paths) else paths
        for p in chosen:
            rows.append({"room_id": room_id, "label": label,
                         "path": str(Path(p).relative_to(ROOT))})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["room_id", "label", "path"])
        w.writeheader()
        w.writerows(rows)

    room_labels = Counter(label_for(r) for r in rooms)
    rir_labels = Counter(r["label"] for r in rows)
    print(f"{'label':16s} {'rooms':>6s} {'RIRs':>6s}")
    for lab in sorted(room_labels):
        print(f"{lab:16s} {room_labels[lab]:6d} {rir_labels[lab]:6d}")
    print(f"\nwrote {len(rows)} RIRs from {len(rooms)} rooms -> {MANIFEST}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-room", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build(a.per_room, a.seed)


if __name__ == "__main__":
    main()
