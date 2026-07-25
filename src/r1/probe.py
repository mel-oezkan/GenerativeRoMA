"""The R1 reconstruction probe: RAE stage-1 decoder on frozen features.

Recipe is a faithful port of RAE stage-1 (arXiv:2510.11690 appendix C.2 /
Table 12, github.com/bytetriper/RAE): L1 + omega_L*LPIPS(VGG, VQGAN-style)
+ omega_G*lambda_adaptive*GAN with a frozen DINO-S/8 StyleGAN-T
discriminator (DiffAug, hinge D / vanilla G), Adam(2e-4, betas (0.5, 0.9)
per paper Table 12 — the released configs use (0.9, 0.95)), cosine decay to
2e-5, warmup 1/16 of training, LPIPS from step 0, disc updates from 6/16,
adversarial loss from 8/16, EMA 0.9978 (EMA weights are the headline eval,
as upstream). The phase boundaries are configured as *epoch fractions* of
the 16-epoch recipe so they follow any step budget.

Deviations forced by 1080 Ti hardware are in the configs, not here: batch 8
(paper 512), 6k steps, fp32, ViT-B decoder (paper default XL does not fit),
320px CO3D pairs instead of 256px ImageNet, noise-tau 0 (RAE trains with
0.8 for diffusion robustness; it would blur the probe).
"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.decoders import RAEDecoder
from src.paths import DINO_S8_CKPT, R1_CACHE_DIR, RESULTS_DIR
from src.rae_gan import (
    DINO_S8_URL,
    DiffAug,
    DinoDisc,
    calculate_adaptive_weight,
    hinge_d_loss,
    vanilla_g_loss,
)
from src.romav2_utils import PairFeatureDataset
from src.splits import get_splits, split_spec
from src.utils import warmup_cosine_lambda, write_run_json
from src.viz.style import agg_figure

# (channels, input stride) per feature tap — structural, not tunable
FEAT_SHAPES = {"dpt": (128, 4), "mv": (1024, 16), "desc": (2048, 16),
               "mvdesc": (3072, 16), "b3": (768, 16)}
DESC_CH = 2048  # mvdesc channel layout: [desc 0:2048, mv 2048:3072]


def cache_dir_for(cfg):
    return Path(cfg.cache_dir) if cfg.cache_dir else R1_CACHE_DIR


def arm_name(cfg):
    """<feature>[_selfpair][_<decoder size>] — the results subdir."""
    arm = f"{cfg.feature}_selfpair" if cfg.self_pair else cfg.feature
    if cfg.decoder.size != "B":  # keep the original B dirs; sweeps get suffixed
        arm = f"{arm}_{cfg.decoder.size}"
    return arm


def out_dir_for(cfg):
    if cfg.out_dir:
        return Path(cfg.out_dir)
    return RESULTS_DIR / split_spec(cfg.split)["out"] / arm_name(cfg)


def phase_steps(cfg):
    """(warmup, disc update start, adversarial start) in optimizer steps."""
    epoch = cfg.optim.steps // cfg.optim.epochs
    return (
        cfg.optim.warmup_epochs * epoch,
        cfg.optim.disc_start_epoch * epoch,
        cfg.optim.gan_start_epoch * epoch,
    )


def build_decoder(feature, size="B", noise_tau=0.0):
    ln_slices = (
        [(0, DESC_CH), (DESC_CH, FEAT_SHAPES["mvdesc"][0])]
        if feature == "mvdesc" else None
    )
    return RAEDecoder(*FEAT_SHAPES[feature], size=size, ln_slices=ln_slices,
                      noise_tau=noise_tau)


def build_datasets(cfg):
    train_pairs, eval_pairs = get_splits(cfg.split, cfg.self_pair)
    cache = cache_dir_for(cfg)
    return (
        PairFeatureDataset(train_pairs, cache, cfg.feature, cfg.size),
        PairFeatureDataset(eval_pairs, cache, cfg.feature, cfg.size),
        train_pairs,
        eval_pairs,
    )


def fetch_batch(ds, idx, device="cuda"):
    feats, targets = zip(*[ds[i] for i in idx])
    return torch.stack(feats).to(device), torch.stack(targets).to(device)


@torch.no_grad()
def update_ema(ema, model, decay):
    ema_params = dict(ema.named_parameters())
    for name, p in model.named_parameters():
        ema_params[name].mul_(decay).add_(p, alpha=1 - decay)


@torch.no_grad()
def evaluate(decoder, eval_ds, lpips_fn, zero_slice=None, device="cuda"):
    was_training = decoder.training
    decoder.eval()
    rows = []
    for i in range(len(eval_ds)):
        f, x = fetch_batch(eval_ds, [i], device)
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


def save_grid(decoder, eval_ds, out_path, n=6, device="cuda"):
    plt, fig, axes = agg_figure(2, n, figsize=(2.2 * n, 4.6))
    with torch.no_grad():
        for col in range(n):
            i = col * (len(eval_ds) // n)
            f, x = fetch_batch(eval_ds, [i], device)
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


def train(cfg):
    import lpips  # eval metric (alex), kept for continuity with v1/v2
    from taming.modules.losses.lpips import LPIPS as TamingLPIPS  # train loss (vgg)

    if not DINO_S8_CKPT.exists():
        raise FileNotFoundError(
            f"DINO-S/8 discriminator weights missing; download {DINO_S8_URL}"
            f" to {DINO_S8_CKPT}"
        )
    device = cfg.device
    steps = cfg.optim.steps
    batch = cfg.optim.batch
    warmup, disc_upd_start, gan_start = phase_steps(cfg)
    out_dir = out_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, eval_ds, train_pairs, eval_pairs = build_datasets(cfg)

    torch.manual_seed(cfg.seed)
    decoder = build_decoder(cfg.feature, cfg.decoder.size, cfg.decoder.noise_tau)
    decoder = decoder.to(device).train()
    if cfg.pipeline:  # L/XL don't fit one 11 GB GPU; split blocks across two
        decoder.pipeline("cuda:1")
    ema = deepcopy(decoder).eval().requires_grad_(False)  # keeps the split
    disc = DinoDisc(DINO_S8_CKPT).to(device).eval()
    disc_aug = DiffAug(prob=1.0, cutout=0.0)

    betas = tuple(cfg.optim.betas)
    opt = torch.optim.Adam(decoder.parameters(), lr=cfg.optim.lr, betas=betas,
                           weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine_lambda(warmup, steps, cfg.optim.min_lr_ratio)
    )
    disc_opt = torch.optim.Adam(disc.parameters(), lr=cfg.optim.lr, betas=betas,
                                weight_decay=0.0)
    # same schedule shape, stepped only on active disc steps (as upstream)
    disc_sched = torch.optim.lr_scheduler.LambdaLR(
        disc_opt, warmup_cosine_lambda(warmup, steps, cfg.optim.min_lr_ratio)
    )

    lpips_train = TamingLPIPS().to(device).eval().requires_grad_(False)
    lpips_eval = lpips.LPIPS(net="alex").to(device)

    ckpt_path = out_dir / "ckpt.pt"
    start = 0
    if ckpt_path.exists():
        # cpu: map_location=device would pin a full extra model+EMA+opt copy
        # on cuda:0 for the whole run (~4.8 GB for size L)
        ckpt = torch.load(ckpt_path, map_location="cpu")
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
        del ckpt
        print(f"resumed at step {start}", flush=True)

    # rng replay so the batch sequence is identical across resumes
    rng = np.random.default_rng(cfg.seed)
    for _ in range(start):
        rng.integers(0, len(train_ds), batch)

    for step in range(start, steps):
        use_gan = step >= gan_start
        train_disc = step >= disc_upd_start

        idx = rng.integers(0, len(train_ds), batch)
        f, x = fetch_batch(train_ds, idx, device)
        x_pm1 = x * 2 - 1

        # --- decoder step (loss assembly as upstream train_stage1.py)
        opt.zero_grad(set_to_none=True)
        xr = decoder(f)
        xr_pm1 = xr * 2 - 1
        l1 = (xr - x).abs().mean()
        lp = lpips_train(x_pm1, xr_pm1).mean()
        recon_total = l1 + cfg.loss.lpips_w * lp
        if use_gan:
            logits_fake = disc(disc_aug.aug(xr_pm1))
            gan_loss = vanilla_g_loss(logits_fake)
            lam = calculate_adaptive_weight(
                recon_total, gan_loss, decoder.last_layer, cfg.loss.max_d_weight
            ).to(device)  # last_layer lives on cuda:1 when pipelined
            loss = recon_total + cfg.loss.disc_w * lam * gan_loss
        else:
            gan_loss = torch.zeros_like(recon_total)
            lam = torch.zeros_like(recon_total)
            loss = recon_total
        loss.backward()
        opt.step()
        sched.step()
        update_ema(ema, decoder, cfg.ema_decay)

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

        if step % cfg.log.every == 0:
            msg = (
                f"[{cfg.feature}] step {step} l1 {l1.item():.4f}"
                f" lpips {lp.item():.4f} gan {gan_loss.item():.4f}"
                f" lam {lam.item():.3f}"
            )
            if d_loss is not None:
                msg += f" d {d_loss.item():.4f}"
            print(msg, flush=True)
        if (step + 1) % cfg.log.ckpt_every == 0 or step + 1 == steps:
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
    metrics = evaluate(ema, eval_ds, lpips_eval, device=device)
    metrics["online"] = evaluate(decoder, eval_ds, lpips_eval, device=device)
    if cfg.feature == "mvdesc":
        # zeroing one source at eval is OOD for the decoder, so a drop only
        # upper-bounds that source's contribution; "mv_zeroed ~= full" is
        # still conclusive (decoder ignored mv)
        metrics["ablations"] = {
            "mv_zeroed": evaluate(ema, eval_ds, lpips_eval, slice(DESC_CH, None),
                                  device),
            "desc_zeroed": evaluate(ema, eval_ds, lpips_eval, slice(0, DESC_CH),
                                    device),
        }
    print(f"[{cfg.feature}] eval: {metrics}", flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_grid(ema, eval_ds, out_dir / "recon_grid.png", device=device)
    write_run_json(
        out_dir,
        {
            **OmegaConf.to_container(cfg, resolve=True),
            "arm": arm_name(cfg),
            "decoder_arch": f"RAE-ViT-{cfg.decoder.size}",
            "warmup": warmup,
            "disc_upd_start": disc_upd_start,
            "gan_start": gan_start,
            "cache_dir": str(cache_dir_for(cfg)),
            "recipe": "RAE stage-1 (arXiv:2510.11690 C.2/Table 12); paper betas"
                      " (0.5,0.9), released configs use (0.9,0.95)",
            "n_train_pairs": len(train_pairs),
            "n_eval_pairs": len(eval_pairs),
        },
    )
    return metrics


def precompute(cfg):
    """Fill the shared feature cache for a split (one process per GPU shard)."""
    from src.romav2_utils import (
        RomaFeatureExtractor,
        cache_file,
        load_romav2_frozen,
        precompute_pair_cache,
    )

    train_pairs, eval_pairs = get_splits(cfg.split, cfg.self_pair)
    pairs = train_pairs + eval_pairs
    if cfg.shard:  # "i/n": disjoint slice so n GPUs can precompute in parallel
        i, n = map(int, cfg.shard.split("/"))
        pairs = pairs[i::n]
    cache = cache_dir_for(cfg)
    extractor = None
    # desc depends only on image A, so the self-pair desc would duplicate the
    # normal-pair sidecars under new keys — skip it
    feature_sets = ([("dpt", "mv"), ("b3",)] if cfg.self_pair
                    else [("dpt", "mv"), ("desc",), ("b3",)])
    for features in feature_sets:
        todo = [
            p for p in pairs if not cache_file(cache, p["key"], features[0]).exists()
        ]
        print(f"{features}: {len(pairs)} pairs total, {len(todo)} to cache",
              flush=True)
        if todo:
            extractor = extractor or RomaFeatureExtractor(load_romav2_frozen())
            precompute_pair_cache(extractor, todo, cache, cfg.size, cfg.device,
                                  features)
    print("cache complete", flush=True)
