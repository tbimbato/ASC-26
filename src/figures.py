"""Figures for the README.

Everything is greyscale: series are told apart by weight, fill and line style
rather than by hue, so the figures survive printing and colour-blind readers.

Needs the BUT corpus under data/raw, so it only runs where the real set is present.

    python src/figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FS = 16000
SPAN = 1.6          # seconds shown; both files decay well inside this
N_FFT, HOP = 512, 128

INK = "#1a1a1a"     # near-black for the foreground series
GREY = "#8a8a8a"    # secondary series
HAIR = "#c8c8c8"    # spines and ticks

PAIR = [
    ("Simulated", ROOT / "data/sim/wav/lecture_room_0328_r0.wav"),
    ("Real (BUT, VUT_FIT_D105)", ROOT / "data/raw/"
     "BUT_ReverbDB_rel_19_06_RIR-Only/VUT_FIT_D105/MicID01/"
     "SpkID06_20170901_S/25/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav"),
]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": HAIR,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.labelsize": 9,
        "axes.labelpad": 7,
        "axes.titlesize": 9.5,
        "axes.titlecolor": INK,
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })


def save(fig, name: str, dpi: int = 200) -> None:
    # GitHub renders a markdown image at its pixel size, so the pixel size is the
    # layout: wide figures get the column, small ones stay small.
    out = RESULTS / f"{name}.png"
    fig.savefig(out, facecolor="white", dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}")


def load(path: Path) -> np.ndarray:
    """Mono, onset-aligned, peak-normalized, cut to SPAN. Same handling as the
    two pipelines, so the picture matches what the models are fed."""
    ir, fs = sf.read(path)
    if ir.ndim > 1:
        ir = ir[:, 0]
    assert fs == FS, f"{path} is {fs} Hz"
    onset = max(0, int(np.argmax(np.abs(ir))) - int(0.0005 * FS))
    ir = ir[onset:onset + int(SPAN * FS)]
    return ir / np.max(np.abs(ir))


def spectrogram(ir: np.ndarray) -> np.ndarray:
    """Log-magnitude STFT in dB below peak."""
    win = np.hanning(N_FFT)
    frames = [np.abs(np.fft.rfft(ir[i:i + N_FFT] * win))
              for i in range(0, len(ir) - N_FFT, HOP)]
    S = np.array(frames).T
    return 20 * np.log10(S / S.max() + 1e-12)


def sim_vs_real() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 5.2),
                             gridspec_kw={"width_ratios": [1, 1.3],
                                          "wspace": 0.16, "hspace": 0.3})
    t = np.arange(int(SPAN * FS)) / FS

    for row, (name, path) in enumerate(PAIR):
        ir = load(path)
        ax = axes[row, 0]
        ax.plot(t[:len(ir)], ir, lw=0.35, color=INK)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xlim(0, SPAN)
        ax.set_ylabel(name)
        ax.set_yticks([-1, 0, 1])

        ax = axes[row, 1]
        im = ax.imshow(spectrogram(ir), origin="lower", aspect="auto",
                       cmap="gray_r", vmin=-90, vmax=0,
                       extent=[0, SPAN, 0, FS / 2000])
        ax.set_ylabel("kHz")
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(HAIR)

    axes[0, 0].set_title("waveform", loc="left")
    axes[0, 1].set_title("spectrogram, dB below peak", loc="left")
    for a in axes[1]:
        a.set_xlabel("seconds")
    for a in axes[0]:
        a.set_xticklabels([])

    cb = fig.colorbar(im, ax=axes[:, 1], pad=0.015, fraction=0.028)
    cb.outline.set_edgecolor(HAIR)
    cb.outline.set_linewidth(0.8)
    fig.suptitle("Lecture room, measured RT60 = 1.10 s in both cases",
                 fontsize=10.5, x=0.075, ha="left", y=1.0)
    save(fig, "sim_vs_real")


def intervals() -> None:
    """Every coarse sim-to-real estimate with its room-clustered interval, on one
    axis. Each neural model contributes its five seeds separately, so the picture
    shows the spread rather than hiding it in a mean."""
    cl = pd.read_csv(RESULTS / "metrics.csv")
    cl = cl[cl["eval"] == "sim2real_coarse"].sort_values("accuracy")
    nn = pd.read_csv(RESULTS / "metrics_nn.csv")
    nn = nn[nn["eval"] == "nn_sim2real_coarse"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = 0
    for _, r in cl.iterrows():
        ax.plot([r.ci_low, r.ci_high], [y, y], lw=2.2, color=INK,
                solid_capstyle="butt")
        ax.plot(r.accuracy, y, "o", ms=6, color=INK, zorder=3)
        ax.text(-0.015, y, r.model, ha="right", va="center", fontsize=9)
        y += 1
    y += 0.7
    for name, g in nn.groupby("model", sort=False):
        for _, r in g.iterrows():
            ax.plot([r.ci_low, r.ci_high], [y, y], lw=0.9, color=GREY,
                    solid_capstyle="butt")
            ax.plot(r.accuracy, y, "o", ms=4.5, mfc="white", mec=GREY,
                    mew=0.9, zorder=3)
            y += 0.42
        ax.text(-0.015, y - 1.26, name, ha="right", va="center", fontsize=9,
                color=GREY)
        y += 0.55

    ax.axvline(0.567, ls=(0, (4, 3)), lw=0.9, color=INK, zorder=1)
    ax.text(0.567, y + 0.15, " majority baseline", fontsize=8.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.9, y + 0.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("coarse sim-to-real accuracy, 95% CI over rooms")
    save(fig, "intervals")


def _table(ax, m, rows, cols, diag=True) -> None:
    """A matrix drawn as a typeset table: hairline rules, no rotated labels, and
    a fill light enough that the numbers stay the thing you read."""
    n_r, n_c = m.shape
    ax.imshow(m, cmap="Greys", vmin=0, vmax=m.max() * 4.5)  # a whisper of shading
    for i in range(n_r):
        for j in range(n_c):
            on_diag = diag and i == j
            ax.text(j, i, m[i, j], ha="center", va="center",
                    fontsize=10.5 if on_diag else 9.5,
                    color=INK if m[i, j] else "#c0c0c0",
                    fontweight="bold" if on_diag else "normal")
    ax.set_xticks(range(n_c), cols, fontsize=8)
    ax.set_yticks(range(n_r), rows, fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0, pad=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for edge in (-0.5, n_r - 0.5):  # booktabs-style top and bottom rule
        ax.axhline(edge, color=INK, lw=0.9, clip_on=False)
    ax.axhline(0.5, color=HAIR, lw=0.7, clip_on=False)
    ax.axhline(1.5, color=HAIR, lw=0.7, clip_on=False)


def confusion() -> None:
    """The best feature-based model against the best net, coarse, side by side.
    Built from the saved predictions so it cannot drift from the metrics."""
    cl = pd.read_csv(RESULTS / "preds.csv")
    cl = cl[(cl["eval"] == "sim2real_coarse") & (cl.model == "XGBoost")]
    nn = pd.read_csv(RESULTS / "preds_nn.csv")
    nn = nn[(nn["eval"] == "nn_sim2real_coarse") & (nn.model == "ResNet18-pt")
            & (nn.seed == 42)]
    short = ["lecture", "small furn.", "stairwell"]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.5),
                             gridspec_kw={"wspace": 0.45})
    for ax, (title, d) in zip(axes, [("XGBoost, six parameters", cl),
                                     ("ResNet18-pt, log-mel", nn)]):
        m = np.zeros((3, 3), int)
        for t, pr in zip(d.y_true, d.y_pred):
            m[t, pr] += 1
        _table(ax, m, short, short)
        ax.set_xlabel(f"{title}, {np.trace(m)}/{m.sum()} correct", fontsize=9,
                      labelpad=10)
    save(fig, "confusion")


def redundancy() -> None:
    """The six parameters against each other, lower triangle only: the upper half
    repeats it and the diagonal is 1 by construction."""
    F = ["rt60", "edt", "c80", "d50", "ts", "drr"]
    d = pd.read_csv(ROOT / "data/real/features.csv").dropna(subset=["label"])
    d = d[d.label.isin(["office", "meeting_room", "lecture_room", "staircase"])]
    c = d[F].corr().abs().values

    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for i in range(1, 6):
        for j in range(i):
            v = c[i, j]
            # shading stretched over the observed range, so DRR reads as the
            # outlier it is instead of blending into the rest
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, lw=0,
                                       color=str(0.97 - (v - 0.5) * 0.52)))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=INK, fontweight="bold" if v > 0.93 else "normal")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(5.5, 0.5)
    ax.set_xticks(range(5), [f.upper() for f in F[:5]], fontsize=7)
    ax.set_yticks(range(1, 6), [f.upper() for f in F[1:]], fontsize=7)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0, pad=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("|r|, 17 real rooms. Bold above 0.93.", fontsize=7.5, labelpad=8)
    save(fig, "feature_redundancy", dpi=150)


if __name__ == "__main__":
    style()
    sim_vs_real()
    intervals()
    confusion()
    redundancy()
