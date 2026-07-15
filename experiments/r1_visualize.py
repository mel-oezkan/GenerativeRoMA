"""R1 probe visualizations across arms (docs/R1.md).

Loads each trained arm's EMA decoder from results/<split dir>/<arm>/ckpt.pt
and renders, on shared eval images:

- viz_recons.png: rows = target + one reconstruction row per arm (annotated
  with the arm's eval PSNR), cols = eval pairs. The qualitative side of the
  desc-vs-mv gap: desc keeps identity/texture, mv/dpt keep layout only.
- viz_decoder_features.png: per arm, PCA->RGB of the decoder's *input*
  feature grid and of token activations after selected blocks, for one eval
  image. Shows where along the decoder depth spatial appearance (re)emerges
  per feature source. PCA is fit per panel (arm x layer) over all shown
  tokens; the 3 components map to RGB, so colors are comparable within a
  panel but not across panels.

CPU-only on purpose: safe to run while training occupies the GPUs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.r1_recon_probe import (
    CACHE_DIR,
    RESULTS_DIR,
    SIZE,
    SPLITS,
    build_decoder,
    get_splits,
)
from src.romav2_utils import PairFeatureDataset

ARMS = ["desc", "mvdesc", "b3", "mv", "dpt"]


def load_ema_decoder(out_root, feature):
    decoder = build_decoder(feature, "B")
    ckpt = torch.load(out_root / feature / "ckpt.pt", map_location="cpu")
    decoder.load_state_dict(ckpt["ema"])
    return decoder.eval()


def pca_rgb(tokens_hw_c):
    """[h, w, C] -> [h, w, 3] via PCA over the h*w tokens, scaled to [0,1]."""
    h, w, c = tokens_hw_c.shape
    x = tokens_hw_c.reshape(-1, c).astype(np.float64)
    x = x - x.mean(0)
    # SVD on centered tokens; top-3 right singular vectors = PCA basis
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    y = x @ vt[:3].T
    lo, hi = np.percentile(y, 2, axis=0), np.percentile(y, 98, axis=0)
    y = np.clip((y - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
    return y.reshape(h, w, 3)


def recon_figure(out_root, eval_sets, decoders, metrics, cols):
    n = len(cols)
    fig, axes = plt.subplots(1 + len(ARMS), n, figsize=(2.1 * n, 2.1 * (1 + len(ARMS))))
    for col_i, ds_i in enumerate(cols):
        _, x = eval_sets[ARMS[0]][ds_i]
        axes[0, col_i].imshow(x.permute(1, 2, 0).numpy())
    axes[0, 0].set_ylabel("target", fontsize=11)
    for row_i, arm in enumerate(ARMS, start=1):
        for col_i, ds_i in enumerate(cols):
            f, _ = eval_sets[arm][ds_i]
            with torch.no_grad():
                xr = decoders[arm](f[None]).clamp(0, 1)[0]
            axes[row_i, col_i].imshow(xr.permute(1, 2, 0).numpy())
        axes[row_i, 0].set_ylabel(
            f"{arm}\n{metrics[arm]['psnr']:.1f} dB", fontsize=11
        )
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    path = out_root / "viz_recons.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def feature_figure(out_root, eval_sets, decoders, ds_i):
    # tap after these fractions of the 12-block ViT-B body
    taps = [0, 3, 7, 11]
    ncols = 2 + len(taps)  # image | input feats | block taps
    fig, axes = plt.subplots(len(ARMS), ncols, figsize=(2.1 * ncols, 2.1 * len(ARMS)))
    for row_i, arm in enumerate(ARMS):
        f, x = eval_sets[arm][ds_i]
        dec = decoders[arm]
        acts = {}
        hooks = [
            dec.blocks[t].register_forward_hook(
                lambda m, i, o, t=t: acts.__setitem__(t, o.detach())
            )
            for t in taps
        ]
        with torch.no_grad():
            xr = dec(f[None]).clamp(0, 1)[0]
        for h in hooks:
            h.remove()
        axes[row_i, 0].imshow(xr.permute(1, 2, 0).numpy())
        axes[row_i, 0].set_ylabel(arm, fontsize=11)
        # raw input feature grid (native stride: 80x80 for dpt, 20x20 else)
        axes[row_i, 1].imshow(pca_rgb(f.permute(1, 2, 0).numpy()))
        for col_i, t in enumerate(taps, start=2):
            tok = acts[t][0, 1:]  # drop CLS
            g = int(tok.shape[0] ** 0.5)
            axes[row_i, col_i].imshow(pca_rgb(tok.reshape(g, g, -1).numpy()))
    titles = ["recon", "input feats"] + [f"block {t}" for t in taps]
    for ax, ti in zip(axes[0], titles):
        ax.set_title(ti, fontsize=11)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    path = out_root / "viz_decoder_features.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="3cat", choices=list(SPLITS))
    ap.add_argument("--n-cols", type=int, default=6)
    ap.add_argument("--feature-idx", type=int, default=0,
                    help="eval index for the decoder-feature figure")
    args = ap.parse_args()

    out_root = RESULTS_DIR / SPLITS[args.split]["out"]
    _, eval_pairs = get_splits(args.split)
    eval_sets = {a: PairFeatureDataset(eval_pairs, CACHE_DIR, a, SIZE) for a in ARMS}
    decoders = {a: load_ema_decoder(out_root, a) for a in ARMS}
    metrics = {
        a: json.loads((out_root / a / "metrics.json").read_text()) for a in ARMS
    }

    n_eval = len(eval_pairs)
    cols = [i * (n_eval // args.n_cols) for i in range(args.n_cols)]
    p1 = recon_figure(out_root, eval_sets, decoders, metrics, cols)
    p2 = feature_figure(out_root, eval_sets, decoders, args.feature_idx)
    print(p1)
    print(p2)


if __name__ == "__main__":
    main()
