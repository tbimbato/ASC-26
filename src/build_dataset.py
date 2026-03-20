from pathlib import Path
from utils import extract_features, ROOM_LABELS
import pandas as pd

df = pd.read_csv("data/processed/rir_paths.csv")

records = []
for i, row in df.iterrows():
    try:
        feats = extract_features(row["path"])
        feats["room_id"] = row["room_id"]
        feats["path"] = str(Path(row["path"]).relative_to(Path.cwd()))
        records.append(feats)
    except Exception as e:
        print(f"SKIP {row['path']}: {e}")

out = pd.DataFrame(records)
out["label"] = out["room_id"].map(ROOM_LABELS)
print(out["label"].value_counts())
out.to_csv("data/processed/features.csv", index=False)
print(f"Done: {len(out)} rows")
