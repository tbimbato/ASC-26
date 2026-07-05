"""Benchmark classical classifiers on room-acoustic features.

Two evaluations:
  A) Stratified 5-fold CV over all classes. RIRs from the same room appear in
     both train and test, so this measures room-dependent performance (an
     optimistic upper bound: the model may recognize the room, not the class).
  B) Leave-one-room-out CV, restricted to classes covered by more than one
     room (office, lecture_room). The model never sees the test room, so this
     measures generalization to unseen rooms. This is the honest number.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FEATURES = ["rt60", "edt", "c80", "d50", "ts", "drr"]


def make_models() -> dict:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", class_weight="balanced", random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(random_state=42, verbosity=0),
    }


def run_cv(df: pd.DataFrame, splitter, groups, eval_name: str,
           metrics: list, feature_cols: list[str]) -> None:
    X = df[feature_cols].values
    y_raw = df["label"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    for model_name, model in make_models().items():
        all_true, all_pred = [], []
        for train_idx, test_idx in splitter.split(X, y, groups):
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

        cm = confusion_matrix(all_true, all_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(f"{model_name} ({eval_name})")
        fig.tight_layout()
        fig.savefig(RESULTS / f"cm_{eval_name}_{model_name}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "features.csv")
    df = df.dropna(subset=["label"] + FEATURES)
    print(df["label"].value_counts(), "\n")

    metrics: list = []

    # A) room-dependent (upper bound)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    run_cv(df, skf, None, "stratified5fold", metrics, FEATURES)

    # B) room-independent, only classes with >= 2 rooms
    rooms_per_class = df.groupby("label")["room_id"].nunique()
    multi = rooms_per_class[rooms_per_class >= 2].index.tolist()
    sub = df[df["label"].isin(multi)].reset_index(drop=True)
    print(f"\nLeave-one-room-out on classes {multi} "
          f"({sub['room_id'].nunique()} rooms, {len(sub)} samples)")
    logo = LeaveOneGroupOut()
    run_cv(sub, logo, sub["room_id"].values, "leave1roomout", metrics, FEATURES)

    out = pd.DataFrame(metrics)
    out.to_csv(RESULTS / "metrics.csv", index=False)
    print(f"\nSaved {RESULTS / 'metrics.csv'} and confusion matrices.")


if __name__ == "__main__":
    main()
