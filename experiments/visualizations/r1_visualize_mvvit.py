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

    python experiments/visualizations/r1_visualize_mvvit.py split=hydrant-full

Configs: configs/figures/r1_mvvit.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.visualizations.r1_visualize import pca_rgb  # noqa: E402
from src.config import hydra_main  # noqa: E402
from src.paths import RESULTS_DIR  # noqa: E402
from src.romav2_utils import img_to_tensor01, load_romav2_frozen  # noqa: E402
from src.splits import get_splits, split_spec  # noqa: E402
from src.viz.style import pyplot  # noqa: E402

plt = pyplot(report_style=False)


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
def run_pair(model, pair, size=320, device="cuda"):
    img_A = img_to_tensor01(Image.open(pair["anchor"]), size).to(device)
    img_B = img_to_tensor01(Image.open(pair["other"]), size).to(device)
    f_A = model.f(img_A)
    f_B = model.f(img_B)
    tap = MvVitTap(model.matcher.mv_vit, (size // 16, size // 16))
    model.matcher(f_A, f_B, img_A=img_A, img_B=img_B, bidirectional=False)
    tap.close()
    # [h, w, 2048] each; index 0 = A, 1 = B (matches tap per-view layout)
    descs = [torch.cat(f, dim=-1)[0].float().cpu() for f in (f_A, f_B)]
    return img_A[0].cpu(), img_B[0].cpu(), descs, tap.acts


def stages_figure(model, pairs, stages, out_path, size=320, device="cuda"):
    ncols = 3 + len(stages)
    fig, axes = plt.subplots(
        len(pairs), ncols, figsize=(1.55 * ncols, 1.55 * len(pairs))
    )
    axes = axes.reshape(len(pairs), ncols)
    for r, pair in enumerate(pairs):
        img_A, img_B, descs, acts = run_pair(model, pair, size, device)
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


def views_figure(model, pair, stages, out_path, size=320, device="cuda"):
    img_A, img_B, descs, acts = run_pair(model, pair, size, device)
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


@hydra_main("figures/r1_mvvit")
def main(cfg):
    out_root = RESULTS_DIR / split_spec(cfg.split)["out"]
    out_root.mkdir(parents=True, exist_ok=True)
    _, eval_pairs = get_splits(cfg.split)
    step = max(1, len(eval_pairs) // cfg.n_pairs)
    pairs = [eval_pairs[i * step] for i in range(cfg.n_pairs)]

    # load model and stages from matcher mv_vit
    model = load_romav2_frozen().to(cfg.device)
    stages = stage_names(len(model.matcher.mv_vit.blocks))

    print(stages_figure(model, pairs, stages, out_root / "viz_mvvit_stages.png",
                        cfg.size, cfg.device))
    print(views_figure(model, eval_pairs[cfg.pair_idx], stages,
                       out_root / "viz_mvvit_views.png", cfg.size, cfg.device))


if __name__ == "__main__":
    main()
