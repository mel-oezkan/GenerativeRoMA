"""fig6 for the depth report: pair-covisibility census + filter funnel.

    python experiments/visualizations/r2_covis_census_fig.py

Reads the census written by r2_pair_covis.py, plots the distribution of
covisible tokens (max over directions, /16 grid) for near vs far train pairs
with the covis floor marked, and appends census stats to the report's
stats.json.

Configs: configs/figures/covis_census.yaml.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from src.co3d_geom import load_frame_index_multi  # noqa: E402
from src.config import hydra_main  # noqa: E402
from src.paths import RESULTS_DIR  # noqa: E402
from src.r2.dataset import build_train_pairs  # noqa: E402
from src.splits import categories_for, covis_cache_for  # noqa: E402
from src.viz.style import GRID_SOFT, INK2, SLOTS, pyplot  # noqa: E402

NEAR_C, FAR_C = SLOTS[0], "#eb6834"


@hydra_main("figures/covis_census")
def main(cfg):
    plt = pyplot(report_style=False)
    floor = cfg.data.covis_floor
    out = RESULTS_DIR / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)

    cc = json.loads(covis_cache_for(cfg.split).read_text())
    index = load_frame_index_multi(categories_for(cfg.split))
    train_pairs, _ = build_train_pairs(index, cfg.split, cfg.data.vq_min,
                                       verbose=False)
    counts = {"near": [], "far": []}
    for p in train_pairs:
        i, j = map(int, p["key"].split("_")[-2:])
        counts["near" if abs(i - j) <= 4 else "far"].append(max(cc[p["key"]]))

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    bins = np.arange(0, cfg.clip + cfg.bin_width, cfg.bin_width)
    for kind, c in [("near", NEAR_C), ("far", FAR_C)]:
        v = np.clip(counts[kind], 0, bins[-1] - 1)
        ax.hist(v, bins=bins, histtype="step", lw=2, color=c,
                label=f"{kind} pairs (n={len(v)})")
    ax.axvline(floor, color=INK2, lw=1.2, ls=":")
    ax.annotate(f"floor = {floor} tokens", (floor, ax.get_ylim()[1]),
                xytext=(6, -14), textcoords="offset points", fontsize=9,
                color=INK2)
    ax.set_xlabel("covisible tokens on the 20×20 grid (best direction,"
                  f" clipped at {cfg.clip})", fontsize=9)
    ax.set_ylabel("train pairs", fontsize=9)
    ax.grid(True, color=GRID_SOFT, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / cfg.out_name, dpi=130)
    plt.close(fig)

    stats_path = out / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    below = {k: int(np.sum(np.array(v) < floor)) for k, v in counts.items()}
    stats["covis_census"] = {
        "floor": floor,
        "split": cfg.split,
        "n_train_after_seq_filters": len(train_pairs),
        "dropped_below_floor": below,
        "n_final": len(train_pairs) - sum(below.values()),
        "median_near": float(np.median(counts["near"])),
        "median_far": float(np.median(counts["far"])),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats["covis_census"], indent=1))


if __name__ == "__main__":
    main()
