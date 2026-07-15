"""fig6 for the depth report: pair-covisibility census + filter funnel.

Reads r2_meta/pair_covis_hydrant.json (r2_pair_covis.py), plots the
distribution of covisible tokens (max over directions, /16 grid) for near
vs far train pairs with the COVIS_FLOOR marked, and appends census stats
to results/r2_depth_report/stats.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.r2_train import COVIS_FLOOR, PAIR_COVIS_CACHE, build_train_pairs
from src.co3d_geom import load_frame_index

OUT = Path(__file__).resolve().parent.parent / "results/r2_depth_report"
NEAR_C, FAR_C = "#2a78d6", "#eb6834"


def main():
    cc = json.loads(PAIR_COVIS_CACHE.read_text())
    train_pairs, _ = build_train_pairs(load_frame_index("hydrant"), verbose=False)
    counts = {"near": [], "far": []}
    for p in train_pairs:
        i, j = map(int, p["key"].split("_")[-2:])
        counts["near" if abs(i - j) <= 4 else "far"].append(max(cc[p["key"]]))

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    bins = np.arange(0, 200, 4)
    for kind, c in [("near", NEAR_C), ("far", FAR_C)]:
        v = np.clip(counts[kind], 0, bins[-1] - 1)
        ax.hist(v, bins=bins, histtype="step", lw=2, color=c,
                label=f"{kind} pairs (n={len(v)})")
    ax.axvline(COVIS_FLOOR, color="#52514e", lw=1.2, ls=":")
    ax.annotate(f"floor = {COVIS_FLOOR} tokens", (COVIS_FLOOR, ax.get_ylim()[1]),
                xytext=(6, -14), textcoords="offset points", fontsize=9,
                color="#52514e")
    ax.set_xlabel("covisible tokens on the 20×20 grid (best direction,"
                  " clipped at 196)", fontsize=9)
    ax.set_ylabel("train pairs", fontsize=9)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_covis_census.png", dpi=130)
    plt.close(fig)

    stats = json.loads((OUT / "stats.json").read_text())
    below = {k: int(np.sum(np.array(v) < COVIS_FLOOR)) for k, v in counts.items()}
    stats["covis_census"] = {
        "floor": COVIS_FLOOR,
        "n_train_after_seq_filters": len(train_pairs),
        "dropped_below_floor": below,
        "n_final": len(train_pairs) - sum(below.values()),
        "median_near": float(np.median(counts["near"])),
        "median_far": float(np.median(counts["far"])),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats["covis_census"], indent=1))


if __name__ == "__main__":
    main()
