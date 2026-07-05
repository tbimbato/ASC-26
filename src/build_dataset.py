"""Compute room-acoustic features for every RIR listed in a manifest.

Works for any manifest (synthetic or real) that has a `path` column pointing to
a WAV RIR. Any other columns (room_id, label, geometry) are carried through, so
the same script serves the synthetic set and the real test set.

    python src/build_dataset.py --manifest data/sim/manifest.csv --out data/sim/features.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from utils import extract_features

ROOT = Path(__file__).resolve().parents[1]


def build(manifest: Path, out: Path) -> None:
    df = pd.read_csv(manifest)
    carry = [c for c in df.columns if c != "path"]

    records = []
    for _, row in df.iterrows():
        try:
            feats = extract_features(row["path"])
        except Exception as e:
            print(f"SKIP {row['path']}: {e}")
            continue
        for c in carry:
            feats[c] = row[c]
        feats["path"] = row["path"]
        records.append(feats)

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"Done: {len(records)} rows -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "sim" / "manifest.csv")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "sim" / "features.csv")
    args = p.parse_args()
    build(args.manifest, args.out)


if __name__ == "__main__":
    main()
