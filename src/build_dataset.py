"""Compute room-acoustic features for every RIR listed in a manifest.

Works for any manifest (synthetic or real) that has a `path` column pointing to
a WAV RIR. Any other columns (room_id, label, geometry) are carried through, so
the same script serves the synthetic set and the real test set.

    python src/build_dataset.py --manifest data/sim/manifest.csv --out data/sim/features.csv
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from utils import extract_features

ROOT = Path(__file__).resolve().parents[1]


def build(manifest: Path, out: Path) -> None:
    df = pd.read_csv(manifest)
    carry = [c for c in df.columns if c != "path"]
    total = len(df)
    start = time.time()

    records = []
    for k, (_, row) in enumerate(df.iterrows(), 1):
        try:
            feats = extract_features(str(ROOT / row["path"]))
        except Exception as e:
            sys.stdout.write("\n")
            print(f"SKIP {row['path']}: {e}")
            continue
        for c in carry:
            feats[c] = row[c]
        feats["path"] = row["path"]
        records.append(feats)
        if k % 20 == 0 or k == total:
            el = time.time() - start
            sys.stdout.write(f"\r  [{k:>5}/{total}] {k / total * 100:5.1f}%  "
                             f"features  {el:4.0f}s   ")
            sys.stdout.flush()
    sys.stdout.write("\n")

    if len(records) < total:
        raise SystemExit(f"{total - len(records)} of {total} RIRs failed to read; "
                         "fix the manifest rather than training on a short set")
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
