"""Qualitative R2-v1 comparison on shared hydrant eval frames (docs/R2.md).

    python experiments/visualizations/r2v1_qualitative_fig.py

Three rows, one per arm of the v1 story:

- joint        : the v1 *joint* matcher's own train-time decoder, co-trained
                 with the matcher from step 0 (12k + GAN)
- match probe  : post-hoc R1 probe decoder on the v1 *match-only* matcher's
                 mv tokens (the collapse)
- desc-recon   : frozen DINOv3 desc through the linear proj + RAE decoder,
                 the train-time "RAE-style" baseline

Comparability note: rows 1 and 3 are *train-time* decoders (r3-4cat eval);
row 2 is a post-hoc probe decoder trained on `hydrant-full`. All probe eval
pairs are a subset of the r3-4cat eval pairs, so the columns are restricted
to those shared hydrant frames and every row sees an identical, held-out
input. The row labels carry each arm's own headline metric, over its own
full eval set, not these columns. Row 2 is therefore a weaker decoder by
construction, not only a weaker feature — the probe axis (joint vs match,
both probes) is the apples-to-apples version of that gap.

Configs: configs/figures/r2v1_qualitative.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.co3d_geom import load_frame_index_multi  # noqa: E402
from src.config import hydra_main  # noqa: E402
from src.paths import FEATS_ROOT, RESULTS_DIR  # noqa: E402
from src.r1.probe import build_decoder  # noqa: E402
from src.r2.dataset import R2PairDataset, as_batch, build_train_pairs  # noqa: E402
from src.r2.model import load_trained_model  # noqa: E402
from src.romav2_utils import PairFeatureDataset  # noqa: E402
from src.splits import categories_for, get_splits  # noqa: E402
from src.viz.io import fig_dir, load_result  # noqa: E402
from src.viz.style import INK, INK2, SURFACE, pyplot  # noqa: E402


def load_ema_decoder(out_dir, device):
    """R1 probe decoder (mv arm, RAE ViT-B), EMA weights = headline eval."""
    decoder = build_decoder("mv", "B")
    ckpt = torch.load(out_dir / "ckpt.pt", map_location="cpu")
    decoder.load_state_dict(ckpt["ema"])
    return decoder.to(device).eval().requires_grad_(False)


def psnr(xr, x):
    return (-10 * torch.log10(((xr - x) ** 2).mean())).item()


@torch.no_grad()
def probe_row(cfg, spec, pairs):
    """Decode the columns through one arm's post-hoc probe decoder."""
    decoder = load_ema_decoder(RESULTS_DIR / spec.run / spec.arm, cfg.device)
    ds = PairFeatureDataset(pairs, FEATS_ROOT / cfg.probe_cache / spec.arm,
                            "mv", cfg.size)
    row = []
    for i in range(len(pairs)):
        f, x = ds[i]
        xr = decoder(f[None].to(cfg.device)).clamp(0, 1)
        row.append((xr[0].cpu(), psnr(xr[0].cpu(), x)))
    del decoder
    torch.cuda.empty_cache()
    return row


@torch.no_grad()
def traintime_row(cfg, spec, batches):
    """Decode through a run's own train-time decoder (co-trained encoder)."""
    model = load_trained_model(
        spec.run, spec.arm, spec.step, with_decoder=True,
        desc_only=spec.get("desc_only", False),
        skip_head=spec.get("skip_head", False), device=cfg.device,
    )
    row = []
    for b in batches:
        xr = model(b["desc_A"], b["desc_B"], b["img_A"],
                   b["img_B"])["recon_A"].clamp(0, 1)
        row.append((xr[0].cpu(), psnr(xr, b["img_A"])))
    del model
    torch.cuda.empty_cache()
    return row


def headline(spec):
    """The row's own reported metric, dug out of its results json."""
    value = load_result(spec.headline)
    for key in spec.headline_key:
        value = value[key]
    return value


@hydra_main("figures/r2v1_qualitative")
@torch.no_grad()
def main(cfg):
    plt = pyplot(report_style=False)

    # shared columns: probe eval pairs, which are a subset of the split's eval
    _, probe_eval = get_splits(cfg.probe_split)
    index = load_frame_index_multi(categories_for(cfg.split))
    _, split_eval = build_train_pairs(index, cfg.split, verbose=False)
    by_key = {p["key"]: i for i, p in enumerate(split_eval)}
    shared = [p for p in probe_eval if p["key"] in by_key]
    assert len(shared) == len(probe_eval), (
        f"{len(probe_eval) - len(shared)} probe eval pairs missing from "
        f"the {cfg.split} eval set — columns would not be shared"
    )

    # evenly spaced across the shared pairs (they are grouped by sequence, so
    # a stride gives distinct hydrant instances rather than neighbours)
    n_cols = cfg.n_cols
    picks = [shared[len(shared) * j // n_cols + len(shared) // (2 * n_cols)]
             for j in range(n_cols)]

    ds = R2PairDataset(split_eval, index, cfg.size)
    batches = [as_batch(ds[by_key[p["key"]]], cfg.device) for p in picks]

    rows = []
    for spec in cfg.rows:
        row = (probe_row(cfg, spec, picks) if spec.kind == "probe"
               else traintime_row(cfg, spec, batches))
        rows.append((spec.label, row, headline(spec)))

    nrows = 1 + len(rows)
    fig, axes = plt.subplots(nrows, n_cols, figsize=(1.75 * n_cols, 1.95 * nrows))
    fig.patch.set_facecolor(SURFACE)

    for col, b in enumerate(batches):
        axes[0, col].imshow(b["img_A"][0].cpu().permute(1, 2, 0).numpy())
        seq = picks[col]["key"].rsplit("_", 2)[0].split("_", 1)[-1]
        axes[0, col].set_title(seq, fontsize=7, color=INK2)

    for r, (_, row, _) in enumerate(rows, start=1):
        for col, (xr, p) in enumerate(row):
            axes[r, col].imshow(xr.permute(1, 2, 0).numpy())
            axes[r, col].set_xlabel(f"{p:.1f} dB", fontsize=7, color=INK2)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    labels = ["target"] + [f"{lab}\n{m:.2f} dB" for lab, _, m in rows]
    for r, lab in enumerate(labels):
        axes[r, 0].set_ylabel(lab, fontsize=8, color=INK, rotation=0,
                              ha="right", va="center", labelpad=8)

    fig.suptitle(cfg.title, fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0.06, 0, 1, 0.95])
    out = fig_dir(cfg.out_dir) / cfg.out_name
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
