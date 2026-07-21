"""Neural baseline, same protocol as the classical side (src/train.py):

  A) nn_insim: train on 80% of the synthetic set (all 11 classes), report
     accuracy on the held-out 20%. The classical side uses 5-fold CV; a
     single stratified split is the standard for a net (5 trainings would
     buy little) and the number is comparable.
  B) nn_sim2real: train on the synthetic overlap classes, test on the real
     held-out rooms. Fine + coarse, coarse via the same COARSE_MAP as
     train.py: predictions re-scored after the fact, never merged before
     training.

The headline comparison is the in-sim -> sim2real DROP of the net vs the
drop of the 6 physical features. Results go to results/metrics_nn.csv
(separate file: train.py rewrites its own metrics.csv wholesale).
"""

import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

from ingest import load_ir
from model import SmallCNN, embed

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "data" / "sim" / "features.csv"    # has path+label, features unused here
REAL = ROOT / "data" / "real" / "features.csv"
RESULTS = ROOT / "results"

OVERLAP_CLASSES = ["office", "meeting_room", "lecture_room", "staircase"]
COARSE_MAP = {  # keep identical to src/train.py
    "office":       "small_furnished",
    "meeting_room": "small_furnished",
    "lecture_room": "lecture_room",
    "staircase":    "staircase",
}

EPOCHS = 60
BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
# Multi-seed: sim2real (esp. coarse) swings ~0.05-0.10 across seeds with only
# 67 real samples, so single-seed numbers are not trustworthy. Report mean+/-std.
SEEDS = [42, 43, 44, 45, 46]

# Augmentation (training batches only, validation stays clean):
# - noise floor: sim IRs are noise-free, real recordings always have a noise
#   floor that eats the decay tail. In the log-mel domain a floor is just an
#   elementwise max with a constant level, drawn per-sample between -70 and
#   -30 dB relative to the peak. This directly attacks the sim-vs-real gap
#   we saw by eye ("simulated IRs way too dry").
# - SpecAugment: zero out a random freq stripe and a random time stripe so
#   the net cannot lean on one narrow cue.
FLOOR_DB = (-70.0, -30.0)
FREQ_MASK = 12   # max masked mel bands
TIME_MASK = 30   # max masked time frames

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def build_xy(df: pd.DataFrame, classes: list[str]):
    """Load + homogenize + embed every file in df. All in memory: 2200
    log-mel images of 64x251 floats is ~140 MB, fine."""
    cidx = {c: i for i, c in enumerate(classes)}
    X, y = [], []
    for n, (path, label) in enumerate(zip(df["path"], df["label"])):
        X.append(embed(load_ir(str(ROOT / path))))
        y.append(cidx[label])
        if (n + 1) % 400 == 0:
            print(f"  embedded {n + 1}/{len(df)}")
    return np.stack(X), np.array(y)


def band_stats(X_tr):
    """Per-mel-band mean/std from the training set only. One number per
    frequency band instead of one global scalar: bands live at very
    different levels and the net should not have to learn that offset."""
    mu = X_tr.mean(axis=(0, 1, 3), keepdims=True)
    sd = X_tr.std(axis=(0, 1, 3), keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Training-time augmentation, in the raw log-mel domain (before
    normalization). See the constants block up top for the reasoning."""
    B = xb.shape[0]
    # noise floor: elementwise max with a per-sample level below the peak
    peak = xb.amax(dim=(1, 2, 3), keepdim=True)
    floor_db = torch.empty(B, 1, 1, 1).uniform_(*FLOOR_DB)
    xb = torch.maximum(xb, peak + floor_db / 10.0)  # log10-power: 10 dB/unit
    # SpecAugment: one freq stripe + one time stripe per sample, filled with
    # the sample mean
    for i in range(B):
        fw = np.random.randint(0, FREQ_MASK + 1)
        tw = np.random.randint(0, TIME_MASK + 1)
        f0 = np.random.randint(0, xb.shape[2] - fw)
        t0 = np.random.randint(0, xb.shape[3] - tw)
        m = xb[i].mean()
        xb[i, :, f0:f0 + fw, :] = m
        xb[i, :, :, t0:t0 + tw] = m
    return xb


def train_model(X_tr, y_tr, X_va, y_va, n_classes, mu, sd, seed) -> torch.nn.Module:
    """Adam + cross-entropy, fixed epochs, keep the best-val-accuracy epoch.
    X_tr and X_va come in raw; augmentation happens on raw training batches,
    then both are normalized with the train-set band stats."""
    torch.manual_seed(seed)
    model = SmallCNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # cosine decay from LR to ~0 over the run: late epochs take smaller steps and
    # settle instead of bouncing around the minimum, so the best-val epoch we
    # keep is a converged model, not a lucky spike on a jittery curve.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = torch.nn.CrossEntropyLoss()
    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    mu_t, sd_t = torch.from_numpy(mu).to(DEVICE), torch.from_numpy(sd).to(DEVICE)
    X_va_n = (X_va - mu) / sd  # val stays clean, normalized once

    best_acc, best_state = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb = augment(X_tr_t[idx].clone()).to(DEVICE)
            xb = (xb - mu_t) / sd_t
            yb = y_tr_t[idx].to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        va_acc = accuracy_score(y_va, predict(model, X_va_n))
        if va_acc > best_acc:
            best_acc, best_state = va_acc, copy.deepcopy(model.state_dict())
        print(f"  epoch {epoch + 1:2d}/{EPOCHS}  val_acc={va_acc:.3f}"
              f"{'  *' if va_acc == best_acc else ''}")

    model.load_state_dict(best_state)
    return model


def predict(model, X) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).to(DEVICE)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def save_confusion(y_true, y_pred, labels, eval_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(f"SmallCNN ({eval_name})")
    fig.tight_layout()
    fig.savefig(RESULTS / f"cm_{eval_name}_SmallCNN.png", dpi=150)
    plt.close(fig)


def score(y_true, y_pred, labels, eval_name, seed, metrics: list,
          save_cm: bool) -> None:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    metrics.append({"eval": eval_name, "model": "SmallCNN", "seed": seed,
                    "accuracy": round(acc, 4), "f1_macro": round(f1m, 4)})
    print(f"[{eval_name}] seed {seed} SmallCNN acc={acc:.3f} f1_macro={f1m:.3f}")
    if save_cm:  # only for the first seed, or the pngs churn 5x
        save_confusion(y_true, y_pred, labels, eval_name)


def main() -> None:
    sim = pd.read_csv(SIM).dropna(subset=["label"])
    real = pd.read_csv(REAL).dropna(subset=["label"])
    print(f"device: {DEVICE}\n")

    # embeddings are seed-independent, compute everything once
    classes = sorted(sim["label"].unique())
    print(f"in-sim: embedding {len(sim)} synthetic RIRs...")
    X, y = build_xy(sim, classes)

    sim_ov = sim[sim["label"].isin(OVERLAP_CLASSES)].reset_index(drop=True)
    real_ov = real[real["label"].isin(OVERLAP_CLASSES)].reset_index(drop=True)
    ov_classes = sorted(OVERLAP_CLASSES)
    print(f"sim2real: embedding {len(sim_ov)} synthetic + "
          f"{len(real_ov)} real RIRs...")
    X_s, y_s = build_xy(sim_ov, ov_classes)
    X_r, y_r = build_xy(real_ov, ov_classes)

    metrics: list = []
    for seed in SEEDS:
        np.random.seed(seed)
        first = seed == SEEDS[0]
        print(f"\n===== seed {seed} =====")

        # ---- A) in-sim, all 11 classes, stratified 80/20 ----
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed)
        mu, sd = band_stats(X_tr)
        model = train_model(X_tr, y_tr, X_va, y_va, len(classes), mu, sd, seed)
        score(y_va, predict(model, (X_va - mu) / sd), classes,
              "nn_insim", seed, metrics, first)

        # ---- B) sim2real, overlap classes, fine + coarse ----
        # small val split from sim only, used to pick the best epoch; the
        # real set is touched exactly once per seed, at the end
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_s, y_s, test_size=0.1, stratify=y_s, random_state=seed)
        mu, sd = band_stats(X_tr)
        model = train_model(X_tr, y_tr, X_va, y_va, len(ov_classes), mu, sd, seed)

        # in-sim number on the same 4-class task, for the in-sim -> sim2real
        # drop (mirrors insim_overlap_5fold on the classical side). Slightly
        # optimistic: this val split also picked the best epoch. Fine here.
        score(y_va, predict(model, (X_va - mu) / sd), ov_classes,
              "nn_insim_overlap", seed, metrics, first)

        y_pred = predict(model, (X_r - mu) / sd)
        score(y_r, y_pred, ov_classes, "nn_sim2real", seed, metrics, first)

        # coarse re-scoring, same predictions
        coarse = sorted(set(COARSE_MAP.values()))
        ci = {c: i for i, c in enumerate(coarse)}
        ct = [ci[COARSE_MAP[ov_classes[i]]] for i in y_r]
        cp = [ci[COARSE_MAP[ov_classes[i]]] for i in y_pred]
        score(ct, cp, coarse, "nn_sim2real_coarse", seed, metrics, first)

    out = pd.DataFrame(metrics)
    out.to_csv(RESULTS / "metrics_nn.csv", index=False)
    print(f"\n===== mean +/- std over {len(SEEDS)} seeds =====")
    agg = out.groupby("eval")[["accuracy", "f1_macro"]].agg(["mean", "std"])
    print(agg.round(3))
    print(f"\nSaved {RESULTS / 'metrics_nn.csv'} and confusion matrices.")


if __name__ == "__main__":
    main()
