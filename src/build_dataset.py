from pathlib import Path

import pandas as pd

from utils import ROOM_LABELS, extract_features

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "processed" / "rir_paths.csv"
OUT_CSV = ROOT / "data" / "processed" / "features.csv"


def main() -> None:
    df = pd.read_csv(IN_CSV)

    records = []
    for _, row in df.iterrows():
        try:
            feats = extract_features(row["path"])
            feats["room_id"] = row["room_id"]
            feats["path"] = str(Path(row["path"]).resolve().relative_to(ROOT))
            records.append(feats)
        except Exception as e:
            print(f"SKIP {row['path']}: {e}")

    out = pd.DataFrame(records)
    out["label"] = out["room_id"].map(ROOM_LABELS)
    print(out["label"].value_counts())
    out.to_csv(OUT_CSV, index=False)
    print(f"Done: {len(out)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
