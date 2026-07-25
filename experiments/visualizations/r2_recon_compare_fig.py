"""Side-by-side recon comparison of the equal-budget train-time decoders
(docs/R2.md "Frozen-DINOv3 baselines"): scratch-joint 12k (deep encoder)
vs desc-joint and desc-recon (frozen DINOv3 desc through a linear proj).

    python experiments/visualizations/r2_recon_compare_fig.py

Decodes the same eval frames (per_cat per category) through each checkpoint
and writes a rows=model, cols=examples grid with per-tile PSNR.

Configs: configs/figures/recon_compare.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.config import hydra_main  # noqa: E402
from src.co3d_geom import load_frame_index_multi  # noqa: E402
from src.r2.dataset import R2PairDataset, as_batch, build_train_pairs  # noqa: E402
from src.r2.model import load_trained_model  # noqa: E402
from src.splits import categories_for  # noqa: E402
from src.viz.io import fig_dir  # noqa: E402
from src.viz.style import INK, INK2, SURFACE, pyplot  # noqa: E402


def pick_examples(pairs, per_cat):
    """`per_cat` evenly spaced eval pairs per category, in category order."""
    by_cat = {}
    for i, p in enumerate(pairs):
        cat = Path(p["anchor"]).parts[-4]
        by_cat.setdefault(cat, []).append(i)
    picks = []
    for cat in sorted(by_cat):
        idxs = by_cat[cat]
        picks += [idxs[len(idxs) * j // per_cat + len(idxs) // (2 * per_cat)]
                  for j in range(per_cat)]
    return picks


@torch.no_grad()
def build_rows(cfg, batches):
    """Per model: [(recon tensor, psnr)] over the sampled frames."""
    rows = []
    for m in cfg.models:
        model = load_trained_model(
            m.run, m.arm, m.step, with_decoder=True, desc_only=m.desc_only,
            skip_head=m.skip_head, device=cfg.device,
        )
        out = []
        for b in batches:
            xr = model(b["desc_A"], b["desc_B"], b["img_A"],
                       b["img_B"])["recon_A"].clamp(0, 1)
            mse = ((xr - b["img_A"]) ** 2).mean()
            out.append((xr[0].cpu(), (-10 * torch.log10(mse)).item()))
        rows.append(out)
        del model
        torch.cuda.empty_cache()
    return rows


@hydra_main("figures/recon_compare")
@torch.no_grad()
def main(cfg):
    plt = pyplot(report_style=False)
    index = load_frame_index_multi(categories_for(cfg.split))
    _, eval_pairs = build_train_pairs(index, cfg.split, verbose=False)
    ds = R2PairDataset(eval_pairs, index, cfg.size)
    picks = pick_examples(eval_pairs, cfg.per_cat)
    batches = [as_batch(ds[i], cfg.device) for i in picks]
    recons = build_rows(cfg, batches)

    n, nrows = len(picks), 1 + len(cfg.models)
    fig, axes = plt.subplots(nrows, n, figsize=(1.75 * n, 1.95 * nrows))
    fig.patch.set_facecolor(SURFACE)
    for col, b in enumerate(batches):
        axes[0, col].imshow(b["img_A"][0].cpu().permute(1, 2, 0).numpy())
        cat = Path(eval_pairs[picks[col]]["anchor"]).parts[-4]
        axes[0, col].set_title(cat, fontsize=8, color=INK2)
    for r, row in enumerate(recons, start=1):
        for col, (xr, psnr) in enumerate(row):
            axes[r, col].imshow(xr.permute(1, 2, 0).numpy())
            axes[r, col].set_xlabel(f"{psnr:.1f} dB", fontsize=7, color=INK2)
    for r in range(nrows):
        for c in range(n):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_visible(False)
    for r, lab in enumerate(["target"] + [m.label for m in cfg.models]):
        axes[r, 0].set_ylabel(lab, fontsize=8, color=INK, rotation=0,
                              ha="right", va="center", labelpad=8)
    fig.suptitle(cfg.title, fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0.04, 0, 1, 0.96])
    out = fig_dir(cfg.out_dir) / cfg.out_name
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
