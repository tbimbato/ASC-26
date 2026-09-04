"""Paired tests between the feature-based and the neural arm.

Both arms are scored on the same 67 RIRs from the same 17 rooms, so the honest
comparison is paired, not two independent accuracies with overlapping intervals.

The test is an exact McNemar on the discordant pairs. It is run at room level,
one majority vote per room, because the several RIRs of one room are the same
acoustic space measured at a few positions and a per-RIR test would treat them
as 67 independent draws. The per-RIR p-values are printed next to it to show
how much that assumption inflates significance.

    python src/significance.py
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVAL = "sim2real_coarse"  # the level where anything separates at all


def room_votes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per room: the true label and the model's majority prediction."""
    g = df.groupby("room_id")
    return pd.DataFrame({
        "y_true": g.y_true.first(),
        "y_pred": g.y_pred.agg(lambda v: v.value_counts().idxmax()),
    })


def mcnemar(hit_a, hit_b) -> tuple[int, int, float]:
    """Exact McNemar: how often A is right where B is wrong, and the reverse."""
    a_only = int((hit_a & ~hit_b).sum())
    b_only = int((~hit_a & hit_b).sum())
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return a_only, b_only, p


def hits(df: pd.DataFrame) -> np.ndarray:
    return (df.y_true == df.y_pred).to_numpy()


def compare(name_a, pred_a, name_b, pred_b) -> None:
    """Same comparison twice: one vote per room, then one per RIR."""
    for level, a, b in [("rooms", room_votes(pred_a), room_votes(pred_b)),
                        ("RIRs", pred_a, pred_b)]:
        ha, hb = hits(a), hits(b)
        wins, losses, p = mcnemar(ha, hb)
        print(f"  {level:5s}  {name_a} {ha.sum()}/{len(ha)} vs {name_b} "
              f"{hb.sum()}/{len(hb)}   discordant {wins}/{losses}   p={p:.3f}")


def main() -> None:
    cl = pd.read_csv(RESULTS / "preds.csv")
    cl = cl[cl["eval"] == EVAL]
    nn = pd.read_csv(RESULTS / "preds_nn.csv")
    nn = nn[nn["eval"] == "nn_" + EVAL]

    # the majority-class baseline as a model that always predicts that class
    base = cl[cl.model == "XGBoost"].copy()
    base["y_pred"] = base.y_true.value_counts().idxmax()

    print(f"{EVAL}: {cl.room_id.nunique()} rooms, {len(base)} RIRs\n")

    print("feature-based vs majority baseline")
    for m in sorted(cl.model.unique()):
        compare(f"{m:12s}", cl[cl.model == m], "baseline", base)

    print("\nfeature-based vs neural, per seed")
    for cm, nm in product(["kNN", "XGBoost"], sorted(nn.model.unique())):
        ps = []
        for seed in sorted(nn.seed.unique()):
            sub = nn[(nn.model == nm) & (nn.seed == seed)]
            _, _, p = mcnemar(hits(room_votes(cl[cl.model == cm])),
                              hits(room_votes(sub)))
            ps.append(p)
        print(f"  rooms  {cm:8s} vs {nm:12s}  p over 5 seeds: "
              f"{min(ps):.3f} to {max(ps):.3f}")


if __name__ == "__main__":
    main()
