"""R1 probe visualizations across arms (docs/R1.md).

    python experiments/visualizations/r1_visualize.py split=hydrant-full

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

Configs: configs/figures/r1_visualize.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import hydra_main  # noqa: E402
from src.paths import R1_CACHE_DIR, RESULTS_DIR  # noqa: E402
from src.r1.probe import build_decoder  # noqa: E402
from src.romav2_utils import PairFeatureDataset  # noqa: E402
from src.splits import get_splits, split_spec  # noqa: E402
from src.viz.io import load_json  # noqa: E402
from src.viz.style import pyplot  # noqa: E402

plt = pyplot(report_style=False)


def load_ema_decoder(out_root, feature, size="B"):
    decoder = build_decoder(feature, size)
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


def strip_axes(axes):
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def recon_figure(out_root, arms, eval_sets, decoders, metrics, cols):
    n = len(cols)
    fig, axes = plt.subplots(1 + len(arms), n,
                             figsize=(2.1 * n, 2.1 * (1 + len(arms))))
    for col_i, ds_i in enumerate(cols):
        _, x = eval_sets[arms[0]][ds_i]
        axes[0, col_i].imshow(x.permute(1, 2, 0).numpy())
    axes[0, 0].set_ylabel("target", fontsize=11)
    for row_i, arm in enumerate(arms, start=1):
        for col_i, ds_i in enumerate(cols):
            f, _ = eval_sets[arm][ds_i]
            with torch.no_grad():
                xr = decoders[arm](f[None]).clamp(0, 1)[0]
            axes[row_i, col_i].imshow(xr.permute(1, 2, 0).numpy())
        axes[row_i, 0].set_ylabel(f"{arm}\n{metrics[arm]['psnr']:.1f} dB",
                                  fontsize=11)
    strip_axes(axes)
    fig.tight_layout()
    path = out_root / "viz_recons.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def feature_figure(out_root, arms, eval_sets, decoders, ds_i, taps):
    ncols = 2 + len(taps)  # image | input feats | block taps
    fig, axes = plt.subplots(len(arms), ncols,
                             figsize=(2.1 * ncols, 2.1 * len(arms)))
    for row_i, arm in enumerate(arms):
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
    for ax, ti in zip(axes[0], ["recon", "input feats"]
                      + [f"block {t}" for t in taps]):
        ax.set_title(ti, fontsize=11)
    strip_axes(axes)
    fig.tight_layout()
    path = out_root / "viz_decoder_features.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


@hydra_main("figures/r1_visualize")
def main(cfg):
    arms = list(cfg.arms)
    out_root = RESULTS_DIR / split_spec(cfg.split)["out"]
    cache = Path(cfg.cache_dir) if cfg.cache_dir else R1_CACHE_DIR
    _, eval_pairs = get_splits(cfg.split)
    eval_sets = {a: PairFeatureDataset(eval_pairs, cache, a, cfg.size)
                 for a in arms}
    decoders = {a: load_ema_decoder(out_root, a, cfg.decoder_size) for a in arms}
    metrics = {a: load_json(out_root / a / "metrics.json") for a in arms}

    cols = [i * (len(eval_pairs) // cfg.n_cols) for i in range(cfg.n_cols)]
    print(recon_figure(out_root, arms, eval_sets, decoders, metrics, cols))
    print(feature_figure(out_root, arms, eval_sets, decoders, cfg.feature_idx,
                         list(cfg.taps)))


if __name__ == "__main__":
    main()
