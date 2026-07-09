"""Assemble the real held-out test set (BUT ReverbDB) for sim-to-real evaluation.

Same room-type vocabulary as the synthetic set (see room_types.py / utils.ROOM_LABELS),
so the two manifests can be used interchangeably by build_dataset.py and train.py.

Subsamples a few RIRs per real room so every room counts equally (BUT has
thousands of measurements in some rooms, one in none), matching the logic
already used for the synthetic set (there, "rooms" are already balanced by
construction).

    python src/real_test.py                  # default: 5 RIRs/room
    python src/real_test.py --per-room 10

MIT reverberation survey and OpenAIR are NOT included (v1 scope: BUT only).
They would add real coverage for the exotic classes (cathedral, gas_tank,
outdoor_patio, outdoor_forest), which currently have zero real ground truth.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from utils import ROOM_LABELS

ROOT = Path(__file__).resolve().parents[1]
BUT_DIR = ROOT / "data" / "raw" / "BUT_ReverbDB_rel_19_06_RIR-Only"
TARGET_NAME = "IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav"

OUT_DIR = ROOT / "data" / "real"
MANIFEST = OUT_DIR / "manifest.csv"


def collect_but_rirs(base_dir: Path) -> dict[str, list[Path]]:
    """room_id -> list of RIR wav paths, for every room in ROOM_LABELS."""
    by_room: dict[str, list[Path]] = {r: [] for r in ROOM_LABELS}
    for wav_path in base_dir.rglob(TARGET_NAME):
        if wav_path.parent.name != "RIR":
            continue
        room_id = wav_path.relative_to(base_dir).parts[0]
        if room_id in by_room:
            by_room[room_id].append(wav_path)
    return by_room


def build(per_room: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    by_room = collect_but_rirs(BUT_DIR)

    rows = []
    for room_id, paths in by_room.items():
        if not paths:
            print(f"  WARN: no RIRs found for {room_id}, skipping")
            continue
        k = min(per_room, len(paths))
        chosen = rng.choice(paths, size=k, replace=False)
        for p in chosen:
            rows.append({
                "room_id": room_id,
                "label": ROOM_LABELS[room_id],
                "path": str(p.relative_to(ROOT)),
            })
        print(f"  {room_id:35s} {ROOM_LABELS[room_id]:16s} {k}/{len(paths)} sampled")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["room_id", "label", "path"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} RIRs from {len(by_room)} rooms -> {MANIFEST}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-room", type=int, default=5,
                   help="RIRs to sample per real room (default 5)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build(a.per_room, a.seed)


if __name__ == "__main__":
    main()
