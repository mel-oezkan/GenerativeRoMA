"""R1: can a small decoder reconstruct images from *frozen* RoMaV2 features?

Baseline endpoint of the reconstructability/matching tradeoff (docs/R1.md):
RoMaV2 stays frozen, only a ConvDecoder is trained on cached pair-conditioned
features. --feature dpt (128ch @ /4, densest), mv (1024ch @ /16, RAE-like),
desc (2048ch @ /16, DINOv3 control), or mvdesc ([desc ‖ mv] 3072ch @ /16
through a 1x1 projection; measures mv's marginal information beyond desc and
is the skip-connection architecture candidate for R2). mvdesc additionally
reports eval-time source ablations (mv/desc channels zeroed) to check the
decoder didn't shortcut to desc alone.
Precompute the shared feature cache once with --precompute-only, then train
one process per feature (cache-only, no RoMaV2 in memory).
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.data import sample_view_pairs, split_seqs
from src.decoders import ConvDecoder, ProjConvDecoder
from src.romav2_utils import (
    PairFeatureDataset,
    RomaFeatureExtractor,
    load_romav2_frozen,
    precompute_pair_cache,
)
from src.utils import warmup_cosine_lambda, write_run_json

SIZE = 320
N_TRAIN_PAIRS_PER_SEQ = 40
N_EVAL_PAIRS_PER_SEQ = 10
STEPS = 6000
BATCH = 8
LR = 2e-4
WARMUP = 200
LPIPS_W = 0.1
CKPT_EVERY = 500
SEED = 0
CACHE_DIR = Path("/visinf/projects_students/dlcv2025_groupZ/romav2_feats/r1_320")
OUT_ROOT = Path(__file__).resolve().parent.parent / "results" / "r1_recon_probe"
FEAT_SHAPES = {"dpt": (128, 4), "mv": (1024, 16), "desc": (2048, 16), "mvdesc": (3072, 16)}
DESC_CH = 2048  # mvdesc channel layout: [desc 0:2048, mv 2048:3072]
DEVICE = "cuda"


def build_decoder(feature):
    in_ch, stride = FEAT_SHAPES[feature]
    if feature == "mvdesc":
        return ProjConvDecoder(in_ch, stride, proj_ch=1024)
    return ConvDecoder(in_ch, stride)


def build_pairs(seqs_by_cat, n_per_seq, seed):
    """Deterministic pairs, alternating near-/far-view conditioning image."""
    pairs, seen = [], set()
    for cat, seqs in seqs_by_cat.items():
        for seq_name, frames in seqs:
            rng = random.Random(f"{seed}_{cat}_{seq_name}")
            for k, s in enumerate(sample_view_pairs(frames, n_per_seq, rng=rng)):
                i, j_near, j_far = s["idx"]
                j, other = (j_near, s["near"]) if k % 2 == 0 else (j_far, s["far"])
                key = f"{cat}_{seq_name}_{i}_{j}"
                if key not in seen:
                    seen.add(key)
                    pairs.append({"key": key, "anchor": s["anchor"], "other": other})
    return pairs


def get_splits():
    eval_seqs, train_seqs = split_seqs()
    train_pairs = build_pairs(train_seqs, N_TRAIN_PAIRS_PER_SEQ, SEED)
    eval_pairs = build_pairs(eval_seqs, N_EVAL_PAIRS_PER_SEQ, SEED)
    return train_pairs, eval_pairs


def fetch_batch(ds, idx):
    feats, targets = zip(*[ds[i] for i in idx])
    return torch.stack(feats).to(DEVICE), torch.stack(targets).to(DEVICE)


@torch.no_grad()
def evaluate(decoder, eval_ds, lpips_fn, zero_slice=None):
    decoder.eval()
    rows = []
    for i in range(len(eval_ds)):
        f, x = fetch_batch(eval_ds, [i])
        if zero_slice is not None:
            f[:, zero_slice] = 0
        xr = decoder(f).clamp(0, 1)
        o = (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        r = (xr[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        rows.append(
            {
                "psnr": peak_signal_noise_ratio(o, r, data_range=255),
                "ssim": structural_similarity(o, r, channel_axis=2, data_range=255),
                "lpips": lpips_fn(x * 2 - 1, xr * 2 - 1).item(),
            }
        )
    decoder.train()
    return {m: float(np.mean([r[m] for r in rows])) for m in ["psnr", "ssim", "lpips"]}


def save_grid(decoder, eval_ds, out_path, n=6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    with torch.no_grad():
        for col in range(n):
            i = col * (len(eval_ds) // n)
            f, x = fetch_batch(eval_ds, [i])
            xr = decoder(f).clamp(0, 1)
            axes[0, col].imshow(x[0].permute(1, 2, 0).cpu().numpy())
            axes[1, col].imshow(xr[0].permute(1, 2, 0).cpu().numpy())
            for row in range(2):
                axes[row, col].axis("off")
    axes[0, 0].set_title("target", loc="left")
    axes[1, 0].set_title("recon", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def train(feature):
    import lpips

    out_dir = OUT_ROOT / feature
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pairs, eval_pairs = get_splits()
    train_ds = PairFeatureDataset(train_pairs, CACHE_DIR, feature, SIZE)
    eval_ds = PairFeatureDataset(eval_pairs, CACHE_DIR, feature, SIZE)

    torch.manual_seed(SEED)
    decoder = build_decoder(feature).to(DEVICE).train()
    opt = torch.optim.Adam(decoder.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine_lambda(WARMUP, STEPS, 0.0)
    )
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    ckpt_path = out_dir / "ckpt.pt"
    start = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        decoder.load_state_dict(ckpt["decoder"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start = ckpt["step"]
        print(f"resumed at step {start}", flush=True)

    # rng replay so the batch sequence is identical across resumes
    rng = np.random.default_rng(SEED)
    for _ in range(start):
        rng.integers(0, len(train_ds), BATCH)

    for step in range(start, STEPS):
        idx = rng.integers(0, len(train_ds), BATCH)
        f, x = fetch_batch(train_ds, idx)
        xr = decoder(f)
        l1 = (xr - x).abs().mean()
        lp = lpips_fn(xr.clamp(0, 1) * 2 - 1, x * 2 - 1).mean()
        loss = l1 + LPIPS_W * lp
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"[{feature}] step {step} l1 {l1.item():.4f} lpips {lp.item():.4f}", flush=True)
        if (step + 1) % CKPT_EVERY == 0 or step + 1 == STEPS:
            torch.save(
                {"decoder": decoder.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "step": step + 1},
                ckpt_path,
            )

    metrics = evaluate(decoder, eval_ds, lpips_fn)
    if feature == "mvdesc":
        # zeroing one source at eval is OOD for the decoder, so a drop only
        # upper-bounds that source's contribution; "mv_zeroed ~= full" is
        # still conclusive (decoder ignored mv)
        metrics["ablations"] = {
            "mv_zeroed": evaluate(decoder, eval_ds, lpips_fn, slice(DESC_CH, None)),
            "desc_zeroed": evaluate(decoder, eval_ds, lpips_fn, slice(0, DESC_CH)),
        }
    print(f"[{feature}] eval: {metrics}", flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_grid(decoder, eval_ds, out_dir / "recon_grid.png")
    write_run_json(
        out_dir,
        {"feature": feature, "size": SIZE, "steps": STEPS, "batch": BATCH,
         "lr": LR, "lpips_w": LPIPS_W, "seed": SEED,
         "n_train_pairs": len(train_pairs), "n_eval_pairs": len(eval_pairs)},
    )


def precompute():
    from src.romav2_utils import cache_file

    train_pairs, eval_pairs = get_splits()
    pairs = train_pairs + eval_pairs
    extractor = None
    for features in [("dpt", "mv"), ("desc",)]:
        todo = [
            p for p in pairs
            if not cache_file(CACHE_DIR, p["key"], features[0]).exists()
        ]
        print(f"{features}: {len(pairs)} pairs total, {len(todo)} to cache", flush=True)
        if todo:
            extractor = extractor or RomaFeatureExtractor(load_romav2_frozen())
            precompute_pair_cache(extractor, todo, CACHE_DIR, SIZE, DEVICE, features)
    print("cache complete", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=list(FEAT_SHAPES))
    ap.add_argument("--precompute-only", action="store_true")
    args = ap.parse_args()
    if args.precompute_only:
        precompute()
        return
    assert args.feature, "--feature required unless --precompute-only"
    train(args.feature)


if __name__ == "__main__":
    main()
