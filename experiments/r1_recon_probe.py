"""R1: can a small decoder reconstruct images from *frozen* RoMaV2 features?

Baseline endpoint of the reconstructability/matching tradeoff (docs/R1.md):
RoMaV2 stays frozen, only an RAEDecoder is trained on cached
pair-conditioned features. Probe v3: decoder + training recipe are a
faithful port of RAE stage-1 (arXiv:2510.11690 appendix C.2 / Table 12,
github.com/bytetriper/RAE): L1 + 1.0*LPIPS(VGG, VQGAN-style) +
0.75*lambda_adaptive*GAN with a frozen DINO-S/8 StyleGAN-T discriminator
(DiffAug, hinge D / vanilla G), Adam(2e-4, betas (0.5, 0.9) per paper Table
12 — the released configs use (0.9, 0.95)), cosine decay to 2e-5, warmup
1/16 of training, LPIPS from step 0, disc updates from 6/16, adversarial
loss on the decoder from 8/16, EMA 0.9978 (EMA weights are the headline
eval, as upstream). Deviations forced by 1080 Ti hardware: batch 8 (paper
512), 6k steps (paper 16 IN1k epochs; phase boundaries keep the paper's
epoch fractions), fp32, ViT-B decoder default (paper default XL does not
fit), 320px CO3D pairs instead of 256px ImageNet, and noise-tau default 0
(RAE trains with 0.8 for diffusion robustness; it would blur the probe).

--feature dpt (128ch @ /4, densest), mv (1024ch @ /16, RAE-like), desc
(2048ch @ /16, DINOv3 control), b3 (mv_vit block-3 tokens, 768ch @ /16;
mid-depth tap where the PCA visualization still shows appearance, before
the b5-b9 collapse), or mvdesc ([desc ‖ mv] 3072ch @ /16; the
embed linear doubles as the mixing projection; measures mv's marginal
information beyond desc and is the skip-connection architecture candidate
for R2). mvdesc additionally reports eval-time source ablations (mv/desc
channels zeroed) to check the decoder didn't shortcut to desc alone.
Precompute the shared feature cache once with --precompute-only, then train
one process per feature (cache-only, no RoMaV2 in memory).

--self-pair conditions the matcher on the anchor itself (A == B) instead of
a second view: same anchors/recipe, cache keys cat_seq_i_i, results in
<feature>_selfpair/. Controls whether cross-view conditioning is what
destroys appearance in mv, or the matcher stages do it even when both
inputs show identical appearance.
"""

import argparse
import random
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.data import sample_view_pairs, split_seqs
from src.decoders import RAEDecoder
from src.rae_gan import (
    DINO_S8_URL,
    DiffAug,
    DinoDisc,
    calculate_adaptive_weight,
    hinge_d_loss,
    vanilla_g_loss,
)
from src.romav2_utils import (
    PairFeatureDataset,
    RomaFeatureExtractor,
    load_romav2_frozen,
    precompute_pair_cache,
)
from src.utils import warmup_cosine_lambda, write_run_json

SIZE = 320
N_EVAL_PAIRS_PER_SEQ = 10
# --split variants; hydrant-full uses every extracted hydrant sequence (726 on
# disk) with fewer pairs/seq to bound precompute time and cache size. Both
# splits shuffle hydrant first with the same seed, so the hydrant eval
# sequences coincide and the arms stay comparable.
SPLITS = {
    "3cat": {"categories": None, "n_train_seqs": 12, "train_pairs_per_seq": 40,
             "out": "r1_recon_probe"},
    "hydrant-full": {"categories": ["hydrant"], "n_train_seqs": None,
                     "train_pairs_per_seq": 10, "out": "r1_recon_probe_hydrant"},
}
STEPS = 6000
BATCH = 8
# RAE Table 12, epoch fractions of the 16-epoch recipe mapped onto STEPS
EPOCH = STEPS // 16
LR = 2e-4
MIN_LR_RATIO = 0.1  # 2e-4 -> 2e-5 cosine
BETAS = (0.5, 0.9)
WARMUP = 1 * EPOCH
DISC_UPD_START = 6 * EPOCH
GAN_START = 8 * EPOCH
LPIPS_W = 1.0  # omega_L
DISC_W = 0.75  # omega_G
MAX_D_WEIGHT = 1e4
EMA_DECAY = 0.9978
CKPT_EVERY = 500
SEED = 0
# cache is shared across splits: pair keys are cat_seq_i_j and the per-seq
# sampling is seed-deterministic, so overlapping pairs are computed once
CACHE_DIR = Path("/visinf/projects_students/dlcv2025_groupZ/romav2_feats/r1_320")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FEAT_SHAPES = {"dpt": (128, 4), "mv": (1024, 16), "desc": (2048, 16),
               "mvdesc": (3072, 16), "b3": (768, 16)}
DESC_CH = 2048  # mvdesc channel layout: [desc 0:2048, mv 2048:3072]
DINO_S8_CKPT = Path.home() / ".cache/torch/hub/checkpoints/dino_deitsmall8_pretrain.pth"
DEVICE = "cuda"


def build_decoder(feature, size, noise_tau=0.0):
    ln_slices = [(0, DESC_CH), (DESC_CH, FEAT_SHAPES["mvdesc"][0])] if feature == "mvdesc" else None
    return RAEDecoder(*FEAT_SHAPES[feature], size=size, ln_slices=ln_slices,
                      noise_tau=noise_tau)


def build_pairs(seqs_by_cat, n_per_seq, seed, self_pair=False):
    """Deterministic pairs, alternating near-/far-view conditioning image.

    self_pair keeps the same sampled anchors but conditions on the anchor
    itself (A == B); anchors drawn twice collapse via the key dedup."""
    pairs, seen = [], set()
    for cat, seqs in seqs_by_cat.items():
        for seq_name, frames in seqs:
            rng = random.Random(f"{seed}_{cat}_{seq_name}")
            for k, s in enumerate(sample_view_pairs(frames, n_per_seq, rng=rng)):
                i, j_near, j_far = s["idx"]
                j, other = (j_near, s["near"]) if k % 2 == 0 else (j_far, s["far"])
                if self_pair:
                    j, other = i, s["anchor"]
                key = f"{cat}_{seq_name}_{i}_{j}"
                if key not in seen:
                    seen.add(key)
                    pairs.append({"key": key, "anchor": s["anchor"], "other": other})
    return pairs


def get_splits(split, self_pair=False):
    cfg = SPLITS[split]
    eval_seqs, train_seqs = split_seqs(
        categories=cfg["categories"], n_train=cfg["n_train_seqs"]
    )
    train_pairs = build_pairs(train_seqs, cfg["train_pairs_per_seq"], SEED, self_pair)
    eval_pairs = build_pairs(eval_seqs, N_EVAL_PAIRS_PER_SEQ, SEED, self_pair)
    return train_pairs, eval_pairs


def fetch_batch(ds, idx):
    feats, targets = zip(*[ds[i] for i in idx])
    return torch.stack(feats).to(DEVICE), torch.stack(targets).to(DEVICE)


@torch.no_grad()
def update_ema(ema, model, decay):
    ema_params = dict(ema.named_parameters())
    for name, p in model.named_parameters():
        ema_params[name].mul_(decay).add_(p, alpha=1 - decay)


@torch.no_grad()
def evaluate(decoder, eval_ds, lpips_fn, zero_slice=None):
    was_training = decoder.training
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
    decoder.train(was_training)
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


def train(feature, size, noise_tau, split, self_pair=False, pipeline=False):
    import lpips  # eval metric (alex), kept for continuity with v1/v2
    from taming.modules.losses.lpips import LPIPS as TamingLPIPS  # training loss (vgg)

    arm = f"{feature}_selfpair" if self_pair else feature
    if size != "B":  # keep the original B dirs; scale sweep gets suffixed arms
        arm = f"{arm}_{size}"
    out_dir = RESULTS_DIR / SPLITS[split]["out"] / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pairs, eval_pairs = get_splits(split, self_pair)
    train_ds = PairFeatureDataset(train_pairs, CACHE_DIR, feature, SIZE)
    eval_ds = PairFeatureDataset(eval_pairs, CACHE_DIR, feature, SIZE)

    torch.manual_seed(SEED)
    decoder = build_decoder(feature, size, noise_tau).to(DEVICE).train()
    if pipeline:  # L/XL don't fit one 11 GB GPU; split blocks across two
        decoder.pipeline("cuda:1")
    ema = deepcopy(decoder).eval().requires_grad_(False)  # keeps the split
    disc = DinoDisc(DINO_S8_CKPT).to(DEVICE).eval()
    disc_aug = DiffAug(prob=1.0, cutout=0.0)

    opt = torch.optim.Adam(decoder.parameters(), lr=LR, betas=BETAS, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine_lambda(WARMUP, STEPS, MIN_LR_RATIO)
    )
    disc_opt = torch.optim.Adam(disc.parameters(), lr=LR, betas=BETAS, weight_decay=0.0)
    # same schedule shape, stepped only on active disc steps (as upstream)
    disc_sched = torch.optim.lr_scheduler.LambdaLR(
        disc_opt, warmup_cosine_lambda(WARMUP, STEPS, MIN_LR_RATIO)
    )

    lpips_train = TamingLPIPS().to(DEVICE).eval().requires_grad_(False)
    lpips_eval = lpips.LPIPS(net="alex").to(DEVICE)

    ckpt_path = out_dir / "ckpt.pt"
    start = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        decoder.load_state_dict(ckpt["decoder"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        disc.load_state_dict(ckpt["disc"])
        disc_opt.load_state_dict(ckpt["disc_opt"])
        disc_sched.load_state_dict(ckpt["disc_sched"])
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
        start = ckpt["step"]
        print(f"resumed at step {start}", flush=True)

    # rng replay so the batch sequence is identical across resumes
    rng = np.random.default_rng(SEED)
    for _ in range(start):
        rng.integers(0, len(train_ds), BATCH)

    for step in range(start, STEPS):
        use_gan = step >= GAN_START
        train_disc = step >= DISC_UPD_START

        idx = rng.integers(0, len(train_ds), BATCH)
        f, x = fetch_batch(train_ds, idx)
        x_pm1 = x * 2 - 1

        # --- decoder step (loss assembly as upstream train_stage1.py)
        opt.zero_grad(set_to_none=True)
        xr = decoder(f)
        xr_pm1 = xr * 2 - 1
        l1 = (xr - x).abs().mean()
        lp = lpips_train(x_pm1, xr_pm1).mean()
        recon_total = l1 + LPIPS_W * lp
        if use_gan:
            logits_fake = disc(disc_aug.aug(xr_pm1))
            gan_loss = vanilla_g_loss(logits_fake)
            lam = calculate_adaptive_weight(
                recon_total, gan_loss, decoder.last_layer, MAX_D_WEIGHT
            ).to(DEVICE)  # last_layer lives on cuda:1 when pipelined
            loss = recon_total + DISC_W * lam * gan_loss
        else:
            gan_loss = torch.zeros_like(recon_total)
            lam = torch.zeros_like(recon_total)
            loss = recon_total
        loss.backward()
        opt.step()
        sched.step()
        update_ema(ema, decoder, EMA_DECAY)

        # --- discriminator step (fresh recon with updated weights, 8-bit
        # discretized fake, DiffAug on both sides, hinge loss)
        d_loss = None
        if train_disc:
            disc.train()
            disc_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                decoder.eval()
                fake = (decoder(f) * 2 - 1).clamp(-1.0, 1.0)
                decoder.train()
                fake = torch.round((fake + 1.0) * 127.5) / 127.5 - 1.0
            logits_fake = disc(disc_aug.aug(fake))
            logits_real = disc(disc_aug.aug(x_pm1))
            d_loss = hinge_d_loss(logits_real, logits_fake)
            d_loss.backward()
            disc_opt.step()
            disc_sched.step()
            disc.eval()

        if step % 100 == 0:
            msg = (
                f"[{feature}] step {step} l1 {l1.item():.4f} lpips {lp.item():.4f}"
                f" gan {gan_loss.item():.4f} lam {lam.item():.3f}"
            )
            if d_loss is not None:
                msg += f" d {d_loss.item():.4f}"
            print(msg, flush=True)
        if (step + 1) % CKPT_EVERY == 0 or step + 1 == STEPS:
            torch.save(
                {"decoder": decoder.state_dict(), "ema": ema.state_dict(),
                 "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "disc": disc.state_dict(), "disc_opt": disc_opt.state_dict(),
                 "disc_sched": disc_sched.state_dict(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(),
                 "step": step + 1},
                ckpt_path,
            )

    # headline metrics from EMA weights (upstream evals the EMA model)
    metrics = evaluate(ema, eval_ds, lpips_eval)
    metrics["online"] = evaluate(decoder, eval_ds, lpips_eval)
    if feature == "mvdesc":
        # zeroing one source at eval is OOD for the decoder, so a drop only
        # upper-bounds that source's contribution; "mv_zeroed ~= full" is
        # still conclusive (decoder ignored mv)
        metrics["ablations"] = {
            "mv_zeroed": evaluate(ema, eval_ds, lpips_eval, slice(DESC_CH, None)),
            "desc_zeroed": evaluate(ema, eval_ds, lpips_eval, slice(0, DESC_CH)),
        }
    print(f"[{feature}] eval: {metrics}", flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_grid(ema, eval_ds, out_dir / "recon_grid.png")
    write_run_json(
        out_dir,
        {"feature": feature, "split": split, "size": SIZE, "steps": STEPS,
         "batch": BATCH,
         "decoder": f"RAE-ViT-{size}", "lr": LR, "min_lr_ratio": MIN_LR_RATIO,
         "betas": BETAS, "warmup": WARMUP, "lpips_w": LPIPS_W,
         "disc_w": DISC_W, "disc_upd_start": DISC_UPD_START,
         "gan_start": GAN_START, "ema_decay": EMA_DECAY,
         "noise_tau": noise_tau, "seed": SEED, "self_pair": self_pair,
         "pipeline": pipeline,
         "recipe": "RAE stage-1 (arXiv:2510.11690 C.2/Table 12); paper betas"
                   " (0.5,0.9), released configs use (0.9,0.95)",
         "n_train_pairs": len(train_pairs), "n_eval_pairs": len(eval_pairs)},
    )


def precompute(split, shard=None, self_pair=False):
    from src.romav2_utils import cache_file

    train_pairs, eval_pairs = get_splits(split, self_pair)
    pairs = train_pairs + eval_pairs
    if shard:  # "i/n": disjoint slice so n GPUs can precompute in parallel
        i, n = map(int, shard.split("/"))
        pairs = pairs[i::n]
    extractor = None
    # desc depends only on image A, so the self-pair desc would duplicate the
    # normal-pair sidecars under new keys — skip it
    feature_sets = ([("dpt", "mv"), ("b3",)] if self_pair
                    else [("dpt", "mv"), ("desc",), ("b3",)])
    for features in feature_sets:
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
    ap.add_argument("--split", default="3cat", choices=list(SPLITS))
    ap.add_argument("--decoder-size", default="B", choices=["S", "B", "L", "XL"],
                    help="RAE Table 13 size; paper default XL exceeds 1080 Ti memory")
    ap.add_argument("--noise-tau", type=float, default=0.0,
                    help="RAE noise-augmented decoding tau (paper 4.3; RAE default"
                         " 0.8, probe default 0)")
    ap.add_argument("--precompute-only", action="store_true")
    ap.add_argument("--shard", default=None,
                    help="i/n slice of the pair list (parallel precompute)")
    ap.add_argument("--pipeline", action="store_true",
                    help="split the decoder across cuda:0/cuda:1 (needed for"
                         " L/XL on 11 GB GPUs); disc/LPIPS stay on cuda:0")
    ap.add_argument("--self-pair", action="store_true",
                    help="condition on the anchor itself (A == B) instead of"
                         " a second view; results go to <feature>_selfpair/")
    args = ap.parse_args()
    if not DINO_S8_CKPT.exists():
        raise FileNotFoundError(f"DINO-S/8 discriminator weights missing; download {DINO_S8_URL} to {DINO_S8_CKPT}")
    if args.precompute_only:
        precompute(args.split, args.shard, args.self_pair)
        return
    assert args.feature, "--feature required unless --precompute-only"
    train(args.feature, args.decoder_size, args.noise_tau, args.split,
          args.self_pair, args.pipeline)


if __name__ == "__main__":
    main()
