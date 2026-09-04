"""Neural baselines, same protocol as the classical side (src/train.py):

  A) nn_insim: train on the synthetic set (all classes), select the epoch on a
     validation split, report on a third split never seen during training.
  B) nn_sim2real: train on the synthetic overlap classes, test on the real
     held-out rooms. Fine + coarse, coarse via the same COARSE_MAP as
     train.py: predictions re-scored after the fact, never merged before
     training. Scored twice, with and without AdaBN.

The comparison is sim2real accuracy against the 6 physical features, over four
architectures (see model.py) on the same log-mel input, split and test set. Beware:
ingest.py truncates to a fixed window while the feature pipeline reads whole files, so
the two arms of the study do not see the same signal (README, Limitations). The
in-sim numbers are reported alongside, but they largely track how well each model
fits the simulator and do not separate the architectures on transfer. Name models
on the command line to train a subset.

Results go to results/metrics_nn.csv (separate file: train.py rewrites its
own metrics.csv wholesale). Rows are merged per model: re-running a model
replaces its rows and leaves the others alone, so the sweep can be run in
pieces.
"""

import copy
import sys
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

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ingest import load_ir
from model import ResNet18Audio, SmallCNN, count_params, embed
from train import majority_baseline, room_bootstrap, room_level_accuracy

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "data" / "sim" / "features.csv"    # has path+label, features unused here
REAL = ROOT / "data" / "real" / "features.csv"
RESULTS = ROOT / "results"
METRICS_CSV = RESULTS / "metrics_nn.csv"
PREDS_CSV = RESULTS / "preds_nn.csv"

OVERLAP_CLASSES = ["office", "meeting_room", "lecture_room", "staircase"]
COARSE_MAP = {  # keep identical to src/train.py
    "office":       "small_furnished",
    "meeting_room": "small_furnished",
    "lecture_room": "lecture_room",
    "staircase":    "staircase",
}

# The sweep: same input, same protocol, different nets. From-scratch nets
# train at 1e-3; the pretrained ResNet fine-tunes at 1e-4 (standard: big
# steps would trash the pretrained features before they can help).
# The epoch budget is per architecture, read off where validation accuracy
# stops moving: the pretrained ResNet is flat by epoch 40, the from-scratch
# CNNs are still climbing at 60. A single budget would either starve the CNNs
# or trade four hours for sixty epochs of ResNet plateau. It cannot be early
# stopping instead: the cosine schedule below anneals over the whole budget, so
# a run cut short keeps a checkpoint that never settled.
MODELS = {
    "CNN-24k":     (lambda n: SmallCNN(n, width=0.5), 1e-3, 120),
    "CNN-94k":     (lambda n: SmallCNN(n, width=1.0), 1e-3, 120),
    "CNN-370k":    (lambda n: SmallCNN(n, width=2.0), 1e-3, 120),
    "ResNet18-pt": (lambda n: ResNet18Audio(n),       1e-4, 60),
}
RUN = sys.argv[1:] or list(MODELS)  # e.g. python train_nn.py CNN-24k CNN-94k
if set(RUN) - set(MODELS):  # embedding runs first, so check before the slow part
    raise SystemExit(f"unknown models {sorted(set(RUN) - set(MODELS))}, "
                     f"pick from {list(MODELS)}")

BATCH = 32
WEIGHT_DECAY = 1e-4
# Seed spread on sim2real (~0.05-0.10 coarse) is as large as the spread between
# the four models, so single-seed numbers cannot separate them.
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
    log-mel images of 64x401 floats is ~500 MB."""
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


def train_model(make_model, lr, epochs, X_tr, y_tr, X_va, y_va, n_classes, mu,
                sd, seed) -> torch.nn.Module:
    """Adam + cross-entropy, fixed epochs, keep the best-val-accuracy epoch.
    X_tr and X_va come in raw; augmentation happens on raw training batches,
    then both are normalized with the train-set band stats."""
    torch.manual_seed(seed)
    model = make_model(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    # cosine decay from lr to ~0 over the run: late epochs take smaller steps and
    # settle instead of bouncing around the minimum, so the best-val epoch we
    # keep has settled, rather than being a spike on a noisy curve.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = torch.nn.CrossEntropyLoss()
    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    mu_t, sd_t = torch.from_numpy(mu).to(DEVICE), torch.from_numpy(sd).to(DEVICE)

    # The selection split is augmented like the training batches, once and fixed.
    # On clean validation the best epoch is the one fitting the dry, silent
    # domain, which is the one that does not transfer.
    torch.manual_seed(seed + 1)
    X_va_n = ((augment(torch.from_numpy(X_va).clone()).numpy()) - mu) / sd

    best_acc, best_state = -1.0, None  # -1 so epoch 1 always sets a state
    for epoch in range(epochs):
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
        print(f"  epoch {epoch + 1:3d}/{epochs}  val_acc={va_acc:.3f}"
              f"{'  *' if va_acc == best_acc else ''}")

    model.load_state_dict(best_state)
    return model


def adabn(model, X, mu, sd, batch=64):
    """Recompute the BatchNorm running statistics on the target domain.

    The stored statistics come from the simulator, where the tail falls to
    digital silence; real recordings sit on a noise floor, so the per-band means
    the network normalizes by are wrong at test time. One forward pass in train
    mode updates them. No labels, no gradients. Modifies the model in place."""
    model.train()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy((X[i:i + batch] - mu) / sd).to(DEVICE)
            model(xb)
    return model


def predict(model, X) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).to(DEVICE)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def save_confusion(y_true, y_pred, labels, eval_name: str, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(f"{model_name} ({eval_name})")
    fig.tight_layout()
    fig.savefig(RESULTS / f"cm_{eval_name}_{model_name}.png", dpi=150)
    plt.close(fig)


def score(y_true, y_pred, labels, eval_name, model_name, seed, metrics: list,
          save_cm: bool, rooms=None, pred_rows: list = None) -> None:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    row = {"eval": eval_name, "model": model_name, "seed": seed,
           "accuracy": round(acc, 4), "f1_macro": round(f1m, 4)}
    line = f"[{eval_name}] seed {seed} {model_name} acc={acc:.3f} f1_macro={f1m:.3f}"
    if rooms is not None:
        # the real set only: same columns train.py reports, or the two arms of
        # the paper end up with confidence intervals on one side and none on the
        # other. The per-sample predictions go to disk so McNemar and anything
        # else a reviewer asks for costs a read, not another six-hour sweep.
        lo, hi = room_bootstrap(y_true, y_pred, rooms)
        row |= {"baseline": round(majority_baseline(y_true), 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "acc_rooms": round(room_level_accuracy(y_true, y_pred, rooms), 4)}
        pred_rows.extend(
            {"eval": eval_name, "model": model_name, "seed": seed,
             "room_id": r, "y_true": int(t), "y_pred": int(pr)}
            for r, t, pr in zip(rooms, y_true, y_pred))
        line += f" rooms={row['acc_rooms']:.3f} ci=[{lo:.3f},{hi:.3f}]"
    metrics.append(row)
    print(line)
    if save_cm:  # first seed only, otherwise the pngs are rewritten five times
        save_confusion(y_true, y_pred, labels, eval_name, model_name)


def merge_rows(new_rows: list, path: Path) -> pd.DataFrame:
    """Replace the re-run (model, seed) rows in path, keep the others."""
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        fresh = set(zip(new["model"], new["seed"]))
        keep = [(m, s) not in fresh for m, s in zip(old["model"], old["seed"])]
        new = pd.concat([old[keep], new], ignore_index=True)
    return new


def already_done(name: str, seed: int) -> bool:
    """A (model, seed) reaches the csv only once its whole block is finished, so
    a single row is enough to know it can be skipped when resuming."""
    if not METRICS_CSV.exists():
        return False
    m = pd.read_csv(METRICS_CSV)
    return bool(((m["model"] == name) & (m["seed"] == seed)).any())


def main() -> None:
    sim = pd.read_csv(SIM).dropna(subset=["label"])
    real = pd.read_csv(REAL).dropna(subset=["label"])
    print(f"device: {DEVICE}\nmodels: {RUN}\n")

    # embeddings are seed- and model-independent, compute everything once
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
    rooms_r = real_ov["room_id"].to_numpy()  # several RIRs per real room

    metrics: list = []
    pred_rows: list = []
    for name in RUN:
        make_model, lr, epochs = MODELS[name]
        print(f"\n########## {name} "
              f"({count_params(make_model(len(classes))):,} params, lr={lr}, "
              f"{epochs} epochs) "
              f"##########")
        for seed in SEEDS:
            if already_done(name, seed):
                print(f"skip {name} seed {seed}, already in metrics_nn.csv")
                continue
            np.random.seed(seed)
            first = seed == SEEDS[0]
            print(f"\n===== {name} seed {seed} =====")

            # ---- A) in-sim, all classes ----
            # Three splits: va picks the best epoch, te is never seen
            # during training and is what gets reported. Scoring on va would be
            # a max-over-60 statistic, not a held-out one.
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=seed)
            X_va, X_te, y_va, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
            mu, sd = band_stats(X_tr)
            model = train_model(make_model, lr, epochs, X_tr, y_tr, X_va, y_va,
                                len(classes), mu, sd, seed)
            score(y_te, predict(model, (X_te - mu) / sd), classes,
                  "nn_insim", name, seed, metrics, first)

            # ---- B) sim2real, overlap classes, fine + coarse ----
            # same three-way split; the real set is touched only at the end
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X_s, y_s, test_size=0.2, stratify=y_s, random_state=seed)
            X_va, X_te, y_va, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
            mu, sd = band_stats(X_tr)
            model = train_model(make_model, lr, epochs, X_tr, y_tr, X_va, y_va,
                                len(ov_classes), mu, sd, seed)

            # in-sim on the same 4-class task, for the drop; held-out split so
            # it is comparable to the classical 5-fold number.
            score(y_te, predict(model, (X_te - mu) / sd), ov_classes,
                  "nn_insim_overlap", name, seed, metrics, first)

            coarse = sorted(set(COARSE_MAP.values()))
            ci = {c: i for i, c in enumerate(coarse)}
            ct = [ci[COARSE_MAP[ov_classes[i]]] for i in y_r]

            # sim2real twice: as-is, then with BN stats adapted to the real set,
            # to separate a genuine failure to transfer from a normalization
            # artifact the classical models have no equivalent of.
            for tag in ["", "_adabn"]:
                if tag:
                    adabn(model, X_r, mu, sd)
                y_pred = predict(model, (X_r - mu) / sd)
                score(y_r, y_pred, ov_classes, "nn_sim2real" + tag, name, seed,
                      metrics, first, rooms_r, pred_rows)
                cp = [ci[COARSE_MAP[ov_classes[i]]] for i in y_pred]
                score(ct, cp, coarse, "nn_sim2real_coarse" + tag, name, seed,
                      metrics, first, rooms_r, pred_rows)

            # The full sweep runs for hours. Writing here means a kill costs the
            # seed in flight and nothing else.
            merge_rows(metrics, METRICS_CSV).to_csv(METRICS_CSV, index=False)
            merge_rows(pred_rows, PREDS_CSV).to_csv(PREDS_CSV, index=False)

    out = pd.read_csv(METRICS_CSV)
    print(f"\n===== mean +/- std over {len(SEEDS)} seeds =====")
    agg = out.groupby(["model", "eval"])[["accuracy", "f1_macro"]].agg(
        ["mean", "std"])
    print(agg.round(3))
    print(f"\nSaved {METRICS_CSV} and confusion matrices.")


if __name__ == "__main__":
    main()
