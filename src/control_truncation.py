"""How much of the feature-based advantage is just seeing the whole file.

The two arms of the study are not fed the same signal: nn/ingest.py fixes every
IR to 3.2 s, cutting the long ones and zero-padding the short ones, while
utils.py estimates the six parameters over the whole file. 17% of the synthetic
training RIRs in the shared classes are cut that way, 61% of the stairwells.

This re-runs the feature-based arm on exactly the signal the network is fed, by
routing every file through ingest.load_ir before extract_features. Three
conditions:

  full          the published pipeline, whole file
  trunc         load_ir output, i.e. what the network sees at evaluation
  trunc_noise   the same plus a noise floor, i.e. what it sees while training

If trunc scores near full, truncation was not carrying the gap between the arms.
If it collapses, part of what the study reads as physics beating a network is a
longer signal beating a shorter one.

Writes results/control_truncation.csv and touches nothing else.

    python src/control_truncation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src" / "nn"))
from ingest import load_ir
from train import (COARSE_MAP, FEATURES, OVERLAP_CLASSES, majority_baseline,
                   make_models, room_bootstrap, room_level_accuracy)
from utils import features_from_ir

OUT = ROOT / "results" / "control_truncation.csv"
NOISE_DB = -50.0  # midpoint of train_nn.FLOOR_DB, applied deterministically


def build(df: pd.DataFrame, noise_db: float | None) -> pd.DataFrame:
    """Features computed from the network's own view of each file."""
    rows = []
    rng = np.random.default_rng(0)
    for i, (_, r) in enumerate(df.iterrows(), 1):
        ir = load_ir(str(ROOT / r["path"]))
        if noise_db is not None:
            # load_ir peak-normalizes, so the floor is relative to unity. The net
            # gets its floor in the log-mel domain; additive noise is the time
            # domain equivalent, not the identical operation.
            ir = ir + rng.normal(0, 10 ** (noise_db / 20), len(ir))
        f = features_from_ir(ir)
        f["room_id"], f["label"] = r["room_id"], r["label"]
        rows.append(f)
        if i % 250 == 0:
            print(f"    {i}/{len(df)}", flush=True)
    return pd.DataFrame(rows).dropna(subset=FEATURES)


def evaluate(sim: pd.DataFrame, real: pd.DataFrame, tag: str, out: list) -> None:
    le = LabelEncoder().fit(sim["label"].values)
    y_tr, y_te = le.transform(sim["label"]), le.transform(real["label"])
    rooms = real["room_id"].values
    coarse = sorted(set(COARSE_MAP.values()))
    ci = {c: i for i, c in enumerate(coarse)}
    ct = [ci[COARSE_MAP[le.classes_[i]]] for i in y_te]

    for name, model in make_models().items():
        sc = StandardScaler()
        model.fit(sc.fit_transform(sim[FEATURES].values), y_tr)
        y_pred = model.predict(sc.transform(real[FEATURES].values))
        cp = [ci[COARSE_MAP[le.classes_[i]]] for i in y_pred]
        for level, t, p in [("fine", y_te, y_pred), ("coarse", ct, cp)]:
            lo, hi = room_bootstrap(t, p, rooms)
            out.append({"condition": tag, "level": level, "model": name,
                        "accuracy": round(accuracy_score(t, p), 4),
                        "f1_macro": round(f1_score(t, p, average="macro"), 4),
                        "baseline": round(majority_baseline(t), 4),
                        "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                        "acc_rooms": round(room_level_accuracy(t, p, rooms), 4)})


def main() -> None:
    sim = pd.read_csv(ROOT / "data/sim/features.csv").dropna(subset=["label"])
    real = pd.read_csv(ROOT / "data/real/features.csv").dropna(subset=["label"])
    sim = sim[sim.label.isin(OVERLAP_CLASSES)]
    real = real[real.label.isin(OVERLAP_CLASSES)]
    print(f"{len(sim)} synthetic, {len(real)} real RIRs, "
          f"{real.room_id.nunique()} rooms")

    out = []
    for tag, noise in [("trunc", None), ("trunc_noise", NOISE_DB)]:
        print(f"\n{tag}: re-extracting features through ingest.load_ir")
        evaluate(build(sim, noise), build(real, noise), tag, out)

    res = pd.DataFrame(out)
    res.to_csv(OUT, index=False)

    published = pd.read_csv(ROOT / "results" / "metrics.csv")
    published = published[published["eval"].isin(["sim2real", "sim2real_coarse"])]
    published["level"] = np.where(
        published["eval"] == "sim2real_coarse", "coarse", "fine")
    published["condition"] = "full"
    table = pd.concat([published[res.columns], res]).pivot_table(
        index=["level", "model"], columns="condition", values="accuracy")
    print("\naccuracy by condition\n")
    print(table[["full", "trunc", "trunc_noise"]].round(3).to_string())
    print(f"\nrooms, coarse:\n")
    print(res[res.level == "coarse"].pivot_table(
        index="model", columns="condition", values="acc_rooms").round(3).to_string())
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
