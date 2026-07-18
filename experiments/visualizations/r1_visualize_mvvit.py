"""Visualize mv_vit features at every stage of the matcher ViT (docs/R1.md).

Runs the frozen RoMaV2 descriptor + matcher on eval pairs and hooks the
mv_vit (MatchTransformer, ViT-B: 12 blocks alternating joint cross-view
attention [even idx] and per-view self-attention with RoPE [odd idx]).
Rendered as PCA->RGB token grids (20x20 at 320px):

- viz_mvvit_stages.png: rows = eval pairs, cols = image A | image B |
  desc input | post-projector | after each block | final output
  (norm + output_projector = exactly the cached "mv" feature). Shows
  *where along the matcher depth* appearance information disappears.
- viz_mvvit_views.png: one pair, rows = view A / view B over the same
  columns — how the two views' token maps co-evolve under the
  alternating joint/self attention.

PCA is fit per panel over its own tokens (top-3 components -> RGB), so
colors are comparable within a panel, not across panels. What to read:
early panels show texture/shading detail inside the object; if later
panels collapse to flat segment-like regions, the appearance needed for
reconstruction is gone by that depth.

Needs a GPU (loads full RoMaV2); features are computed fresh, the R1
cache is not touched.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from experiments.r1_recon_probe import RESULTS_DIR, SIZE, SPLITS, get_splits
from experiments.r1_visualize import pca_rgb
from src.romav2_utils import img_to_tensor01, load_romav2_frozen

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MvVitTap:
    """Hooks mv_vit stages; every capture is normalized to per-view grids
    [2, h, w, D] (view 0 = A). Handles both token layouts the alternating
    blocks use: joint [B, 2*h*w, D] and per-view [(B*2), h*w, D].

    MV is a special case where we need to differentiate between the two views.
    We have alternating layers of joint attention and self attention.
    More information in paper: https://arxiv.org/abs/2511.15706
    """

    def __init__(self, mv_vit, hw):
        self.hw = hw
        self.acts = {}
        self.hooks = [
            mv_vit.projector.register_forward_hook(self._save("proj")),
            mv_vit.output_projector.register_forward_hook(self._save("out")),
        ] + [
            blk.register_forward_hook(self._save(f"block{i}"))
            for i, blk in enumerate(mv_vit.blocks)
        ]

    def _save(self, name):
        """Helper function to create a hook that saves the output of a module"""
        def hook(m, inp, out):
            h, w = self.hw
            self.acts[name] = out.detach().float().reshape(2, h, w, -1).cpu()

        return hook

    def close(self):
        for h in self.hooks:
            h.remove()


def stage_names(n_blocks):
    names = [("proj", "proj in")]
    for i in range(n_blocks):
        kind = "self" if i % 2 == 1 else "joint"
        names.append((f"block{i}", f"b{i} {kind}"))
    names.append(("out", "out (mv)"))
    return names


def show(ax, img, ylabel=None, title=None):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=9)


@torch.no_grad()
def run_pair(model, pair):
    img_A = img_to_tensor01(Image.open(pair["anchor"]), SIZE).to(DEVICE)
    img_B = img_to_tensor01(Image.open(pair["other"]), SIZE).to(DEVICE)
    f_A = model.f(img_A)
    f_B = model.f(img_B)
    tap = MvVitTap(model.matcher.mv_vit, (SIZE // 16, SIZE // 16))
    model.matcher(f_A, f_B, img_A=img_A, img_B=img_B, bidirectional=False)
    tap.close()
    # [h, w, 2048] each; index 0 = A, 1 = B (matches tap per-view layout)
    descs = [torch.cat(f, dim=-1)[0].float().cpu() for f in (f_A, f_B)]
    return img_A[0].cpu(), img_B[0].cpu(), descs, tap.acts


def stages_figure(model, pairs, stages, out_path):
    ncols = 3 + len(stages)
    fig, axes = plt.subplots(
        len(pairs), ncols, figsize=(1.55 * ncols, 1.55 * len(pairs))
    )
    axes = axes.reshape(len(pairs), ncols)
    for r, pair in enumerate(pairs):
        img_A, img_B, descs, acts = run_pair(model, pair)
        show(axes[r, 0], img_A.permute(1, 2, 0), ylabel=pair["key"].split("_")[0])
        show(axes[r, 1], img_B.permute(1, 2, 0))
        show(axes[r, 2], pca_rgb(descs[0].numpy()))
        for c, (key, _) in enumerate(stages, start=3):
            show(axes[r, c], pca_rgb(acts[key][0].numpy()))
    titles = ["image A", "image B", "desc"] + [t for _, t in stages]
    for ax, ti in zip(axes[0], titles):
        ax.set_title(ti, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def views_figure(model, pair, stages, out_path):
    img_A, img_B, descs, acts = run_pair(model, pair)
    ncols = 2 + len(stages)
    fig, axes = plt.subplots(2, ncols, figsize=(1.55 * ncols, 1.55 * 2))
    for v, (img, name) in enumerate([(img_A, "view A"), (img_B, "view B")]):
        show(axes[v, 0], img.permute(1, 2, 0), ylabel=name)
        show(axes[v, 1], pca_rgb(descs[v].numpy()))
        for c, (key, _) in enumerate(stages, start=2):
            show(axes[v, c], pca_rgb(acts[key][v].numpy()))
    titles = ["image", "desc"] + [t for _, t in stages]
    for ax, ti in zip(axes[0], titles):
        ax.set_title(ti, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="3cat", choices=list(SPLITS))
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument(
        "--pair-idx", type=int, default=0, help="eval index for the two-view figure"
    )
    args = ap.parse_args()

    out_root = RESULTS_DIR / SPLITS[args.split]["out"]
    out_root.mkdir(parents=True, exist_ok=True)
    _, eval_pairs = get_splits(args.split)
    step = max(1, len(eval_pairs) // args.n_pairs)
    pairs = [eval_pairs[i * step] for i in range(args.n_pairs)]

    # load model and stages from matcher mv_vit
    model = load_romav2_frozen().to(DEVICE)
    stages = stage_names(len(model.matcher.mv_vit.blocks))

    p1 = stages_figure(model, pairs, stages, out_root / "viz_mvvit_stages.png")
    p2 = views_figure(
        model, eval_pairs[args.pair_idx], stages, out_root / "viz_mvvit_views.png"
    )
    print(p1)
    print(p2)


if __name__ == "__main__":
    main()
