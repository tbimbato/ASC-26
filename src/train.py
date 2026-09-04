"""Benchmark classical classifiers on room-acoustic features.

Two evaluations:
  A) Stratified 5-fold CV on the synthetic set, all 10 classes. Each
     synthetic room_id is a distinct simulated room sampled once, so there is
     no repeated-room leak here: this is already a room-independent estimate
     of in-sim performance.
  B) Sim-to-real: train on the synthetic set restricted to the classes that
     overlap with the real set (BUT + AIR + ACE: office, meeting_room,
     lecture_room, staircase), test on the real held-out rooms. Reported at
     two levels: fine (functional labels as-is) and coarse (labels collapsed
     to acoustic archetypes AFTER prediction, same model, same predictions).
     The coarse map is defined a-priori from physics: a small office and a
     small meeting room are the same acoustic object.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FEATURES = ["rt60", "edt", "c80", "d50", "ts", "drr"]
OVERLAP_CLASSES = ["office", "meeting_room", "lecture_room", "staircase"]

# Acoustic archetypes: functional labels that describe the same physical
# space collapse into one class. Office and small meeting room share size,
# furniture and absorption; the features cannot (and should not) tell them
# apart. Lecture room and staircase keep their own real acoustic signature.
COARSE_MAP = {
    "office":       "small_furnished",
    "meeting_room": "small_furnished",
    "lecture_room": "lecture_room",
    "staircase":    "staircase",
}


def make_models() -> dict:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", class_weight="balanced", random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(random_state=42, verbosity=0),
    }


def save_confusion(y_true, y_pred, labels, model_name: str, eval_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(f"{model_name} ({eval_name})")
    fig.tight_layout()
    fig.savefig(RESULTS / f"cm_{eval_name}_{model_name}.png", dpi=150)
    plt.close(fig)


def run_cv(df: pd.DataFrame, eval_name: str, metrics: list,
           feature_cols: list[str]) -> None:
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in make_models().items():
        all_true, all_pred = [], []
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model.fit(X_train, y[train_idx])
            all_true.extend(y[test_idx])
            all_pred.extend(model.predict(X_test))

        acc = accuracy_score(all_true, all_pred)
        f1m = f1_score(all_true, all_pred, average="macro")
        metrics.append({"eval": eval_name, "model": model_name,
                        "accuracy": round(acc, 4), "f1_macro": round(f1m, 4)})
        print(f"[{eval_name}] {model_name:12s} acc={acc:.3f} f1_macro={f1m:.3f}")
        save_confusion(all_true, all_pred, le.classes_, model_name, eval_name)


def majority_baseline(y) -> float:
    """Accuracy of always predicting the most common class."""
    y = np.asarray(y)
    return np.bincount(y).max() / len(y)


def room_bootstrap(y_true, y_pred, rooms, n_boot=5000, seed=0):
    """95% CI on accuracy, resampling rooms rather than RIRs.

    The several RIRs of one room are the same acoustic space measured at a few
    positions, not independent draws, so a CI over the 67 samples is far too
    narrow. There are 17 rooms."""
    y_true, y_pred, rooms = np.asarray(y_true), np.asarray(y_pred), np.asarray(rooms)
    uniq = np.unique(rooms)
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(rooms == r) for r in picked])
        accs.append((y_true[idx] == y_pred[idx]).mean())
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def room_level_accuracy(y_true, y_pred, rooms) -> float:
    """One vote per room, by majority of its RIRs."""
    y_true, y_pred, rooms = np.asarray(y_true), np.asarray(y_pred), np.asarray(rooms)
    hits = 0
    for r in np.unique(rooms):
        m = rooms == r
        vote = np.bincount(y_pred[m]).argmax()
        hits += vote == y_true[m][0]
    return hits / len(np.unique(rooms))


def run_holdout(train_df: pd.DataFrame, test_df: pd.DataFrame, eval_name: str,
                metrics: list, feature_cols: list[str], save_cm: bool = True) -> None:
    le = LabelEncoder()
    le.fit(train_df["label"].values)
    y_train = le.transform(train_df["label"].values)
    y_test = le.transform(test_df["label"].values)
    X_train_raw = train_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values
    rooms = test_df["room_id"].values

    coarse_labels = sorted(set(COARSE_MAP.values()))
    cidx = {c: i for i, c in enumerate(coarse_labels)}
    ct = [cidx[COARSE_MAP[le.classes_[i]]] for i in y_test]

    print(f"  baseline (majority class): fine {majority_baseline(y_test):.3f}  "
          f"coarse {majority_baseline(ct):.3f}   [{len(np.unique(rooms))} rooms, "
          f"{len(y_test)} RIRs]")

    for model_name, model in make_models().items():
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        lo, hi = room_bootstrap(y_test, y_pred, rooms)
        metrics.append({"eval": eval_name, "model": model_name,
                        "accuracy": round(acc, 4), "f1_macro": round(f1m, 4),
                        "baseline": round(majority_baseline(y_test), 4),
                        "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                        "acc_rooms": round(room_level_accuracy(y_test, y_pred, rooms), 4)})
        print(f"[{eval_name}] {model_name:12s} acc={acc:.3f} [{lo:.3f},{hi:.3f}] "
              f"f1={f1m:.3f} rooms={room_level_accuracy(y_test, y_pred, rooms):.3f}")
        if save_cm:
            save_confusion(y_test, y_pred, le.classes_, model_name, eval_name)

        # Coarse re-scoring: identical model, identical predictions, labels
        # collapsed to acoustic archetypes after the fact. Measures how much
        # of the fine error is intra-archetype (office vs meeting room).
        cp = [cidx[COARSE_MAP[le.classes_[i]]] for i in y_pred]
        acc_c = accuracy_score(ct, cp)
        f1_c = f1_score(ct, cp, average="macro")
        lo_c, hi_c = room_bootstrap(ct, cp, rooms)
        metrics.append({"eval": eval_name + "_coarse", "model": model_name,
                        "accuracy": round(acc_c, 4), "f1_macro": round(f1_c, 4),
                        "baseline": round(majority_baseline(ct), 4),
                        "ci_low": round(lo_c, 4), "ci_high": round(hi_c, 4),
                        "acc_rooms": round(room_level_accuracy(ct, cp, rooms), 4)})
        print(f"[{eval_name}_coarse] {model_name:12s} acc={acc_c:.3f} "
              f"[{lo_c:.3f},{hi_c:.3f}] f1={f1_c:.3f} "
              f"rooms={room_level_accuracy(ct, cp, rooms):.3f}")
        if save_cm:
            save_confusion(ct, cp, coarse_labels, model_name, eval_name + "_coarse")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sim-features", type=Path, default=ROOT / "data" / "sim" / "features.csv")
    p.add_argument("--real-features", type=Path, default=ROOT / "data" / "real" / "features.csv")
    args = p.parse_args()

    sim = pd.read_csv(args.sim_features).dropna(subset=["label"] + FEATURES)
    real = pd.read_csv(args.real_features).dropna(subset=["label"] + FEATURES)
    print(sim["label"].value_counts(), "\n")

    metrics: list = []

    # A) in-sim, all classes, already room-independent (one RIR per simulated room)
    run_cv(sim, "insim_5fold", metrics, FEATURES)

    # B) sim-to-real: train on synthetic overlap classes, test on real held-out rooms
    sim_overlap = sim[sim["label"].isin(OVERLAP_CLASSES)].reset_index(drop=True)
    real_overlap = real[real["label"].isin(OVERLAP_CLASSES)].reset_index(drop=True)

    # in-sim on the same 4 overlap classes, so the in-sim -> sim2real drop is
    # computed on the same task (the 10-class number is not comparable)
    run_cv(sim_overlap, "insim_overlap_5fold", metrics, FEATURES)
    print(f"\nSim-to-real on classes {OVERLAP_CLASSES}: "
          f"{len(sim_overlap)} synthetic train rooms, "
          f"{real_overlap['room_id'].nunique()} real test rooms "
          f"({len(real_overlap)} samples)")
    run_holdout(sim_overlap, real_overlap, "sim2real", metrics, FEATURES)

    # Which of the six actually carry the transfer, and whether one is enough.
    print("\nfeature ablation")
    run_holdout(sim_overlap, real_overlap, "abl_rt60_only", metrics, ["rt60"],
                save_cm=False)
    for f in FEATURES:
        rest = [c for c in FEATURES if c != f]
        run_holdout(sim_overlap, real_overlap, f"abl_no_{f}", metrics, rest,
                    save_cm=False)

    out = pd.DataFrame(metrics)
    out.to_csv(RESULTS / "metrics.csv", index=False)
    print(f"\nSaved {RESULTS / 'metrics.csv'} and confusion matrices.")


if __name__ == "__main__":
    main()
