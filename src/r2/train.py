"""R2/R3 training loop: joint matching + reconstruction on CO3D pairs.

Arms (cfg.arm):
  match  matching losses only (lam_rec forced to 0)
  joint  matching + lam_rec * recon
  recon  recon only (matching-loss weights zeroed)

Everything the loop needs comes from the config; the only implicit inputs
are the caches on disk (per-frame desc, covisibility census).
"""

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.co3d_geom import load_frame_index_multi
from src.paths import DINO_S8_CKPT, RESULTS_DIR
from src.r2.arms import arm_flags
from src.r2.dataset import (
    R2PairDataset,
    collate,
    filtered_train_pairs,
    to_device,
)
from src.r2.losses import matching_loss, recon_loss
from src.r2.metrics import evaluate
from src.r2.model import R2Model, load_pretrained_matcher
from src.r2.viz import save_panels, save_recon_grid
from src.splits import categories_for
from src.utils import warmup_cosine_lambda, write_run_json


def out_dir_for(cfg):
    """results/<run>/<arm>, or <out_dir>/<arm> when overridden (smoke runs)."""
    root = Path(cfg.out_dir) if cfg.out_dir else RESULTS_DIR / cfg.run
    return root / cfg.arm


def build_datasets(cfg, verbose=True):
    """(train_ds, eval_ds) under the configured filter pipeline."""
    index = load_frame_index_multi(categories_for(cfg.split))
    train_pairs, eval_pairs = filtered_train_pairs(
        index, cfg.split, cfg.data.vq_min, cfg.data.covis_floor, verbose
    )
    if verbose:
        print(
            f"final: {len(train_pairs)} train / {len(eval_pairs)} eval pairs",
            flush=True,
        )
    ds = lambda pairs: R2PairDataset(  # noqa: E731
        pairs, index, cfg.size, cfg.data.desc_cache
    )
    return ds(train_pairs), ds(eval_pairs)


def build_model(cfg, flags):
    model = R2Model(
        flags.with_decoder,
        desc_only=flags.desc_only,
        skip_head=flags.skip_head,
        decoder_size=cfg.model.decoder_size,
    ).to(cfg.device).train()
    if cfg.model.init == "pretrained":
        load_pretrained_matcher(model)
    return model


def build_optimizer(cfg, model):
    lr = cfg.optim.lr
    m_lr = cfg.optim.matcher_lr if cfg.optim.matcher_lr is not None else lr
    groups = [
        {"params": model.matcher.parameters(),
         "weight_decay": cfg.optim.wd_matcher, "lr": m_lr}
    ]
    if model.decoder is not None:
        groups.append(
            {"params": model.decoder.parameters(), "weight_decay": 0.0, "lr": lr}
        )
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine_lambda(cfg.optim.warmup, cfg.optim.steps,
                                  cfg.optim.min_lr_ratio)
    )
    return opt, sched, m_lr


def build_gan(cfg, model):
    """Train-time GAN: RAE stage-1 recipe (src/rae_gan.py) with delayed phase
    starts. Deviation from upstream: the discriminator trains on the
    *detached generator-pass* recons (VQGAN-style) instead of a fresh
    post-update forward — a fresh forward would re-run the matcher, which
    dominates step cost.
    """
    from src.rae_gan import DiffAug, DinoDisc

    disc = DinoDisc(DINO_S8_CKPT).to(cfg.device).eval()
    disc_opt = torch.optim.Adam(disc.parameters(), lr=cfg.optim.lr, betas=(0.5, 0.9))
    disc_sched = torch.optim.lr_scheduler.LambdaLR(
        disc_opt, warmup_cosine_lambda(cfg.optim.warmup, cfg.optim.steps,
                                       cfg.optim.min_lr_ratio)
    )
    return disc, DiffAug(prob=1.0, cutout=0.0), disc_opt, disc_sched


def train(cfg):
    from taming.modules.losses.lpips import LPIPS as TamingLPIPS

    flags = arm_flags(cfg)
    device = cfg.device
    out_dir = out_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, eval_ds = build_datasets(cfg)

    torch.manual_seed(cfg.seed)
    model = build_model(cfg, flags)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"arm {cfg.arm}: {n_params:.1f}M params (decoder: {flags.with_decoder})",
        flush=True,
    )

    opt, sched, m_lr = build_optimizer(cfg, model)
    lpips_fn = None
    if flags.with_decoder:
        lpips_fn = TamingLPIPS().to(device).eval().requires_grad_(False)

    steps = cfg.optim.steps
    disc = disc_aug = disc_opt = disc_sched = None
    disc_start = int(cfg.gan.disc_start_frac * steps)
    gan_start = int(cfg.gan.gan_start_frac * steps)
    if flags.gan:
        from src.rae_gan import (
            calculate_adaptive_weight,
            hinge_d_loss,
            vanilla_g_loss,
        )

        disc, disc_aug, disc_opt, disc_sched = build_gan(cfg, model)
        print(
            f"train-time GAN: disc updates from step {disc_start}, "
            f"adversarial loss from step {gan_start}",
            flush=True,
        )

    # pre-drawn index sequence -> exact resume without replaying data loads
    batch_size, accum = cfg.optim.batch, cfg.optim.accum
    g = torch.Generator().manual_seed(cfg.seed)
    idx_all = torch.randint(len(train_ds), (steps * batch_size * accum,), generator=g)

    ckpt_path = out_dir / "ckpt.pt"
    start = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        if disc is not None and "disc" in ckpt:
            disc.load_state_dict(ckpt["disc"])
            disc_opt.load_state_dict(ckpt["disc_opt"])
            disc_sched.load_state_dict(ckpt["disc_sched"])
        start = ckpt["step"]
        print(f"resumed at step {start}", flush=True)

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_sampler=idx_all[start * batch_size * accum :]
        .reshape(-1, batch_size)
        .tolist(),
        num_workers=cfg.data.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    it = iter(loader)

    n_skipped = 0
    zero = torch.zeros((), device=device)
    lam_rec = flags.lam_rec
    for step in range(start, steps):
        use_gan = disc is not None and step >= gan_start
        train_disc = disc is not None and step >= disc_start
        opt.zero_grad(set_to_none=True)
        # NaN guard: a single bad micro-batch (e.g. inf in fp16 depth GT)
        # must not poison the weights — skip the whole optimizer step but
        # keep the schedules
        bad = None
        fakes, reals = [], []
        for _ in range(accum):
            b = to_device(next(it), device)
            preds = model(b["desc_A"], b["desc_B"], b["img_A"], b["img_B"])

            l_attn = l_warp = l_conf = l1 = lp = l_gan = lam = recon_total = zero
            if flags.use_matching:
                l_attn, l_warp, l_conf = matching_loss(preds, b)
            if flags.with_decoder:
                l1, lp, recon_total = recon_loss(preds, b, lpips_fn, cfg.loss.lpips_w)
            if use_gan:
                l_gan = (
                    vanilla_g_loss(disc(disc_aug.aug(preds["recon_A"] * 2 - 1)))
                    + vanilla_g_loss(disc(disc_aug.aug(preds["recon_B"] * 2 - 1)))
                ) / 2
                lam = calculate_adaptive_weight(
                    recon_total, l_gan, model.decoder.last_layer, cfg.gan.max_d_weight
                )
            loss = (
                cfg.loss.w_attn * l_attn
                + cfg.loss.w_warp * l_warp
                + cfg.loss.w_conf * l_conf
                + lam_rec * (recon_total + cfg.gan.disc_w * lam * l_gan)
            ) / accum
            if not torch.isfinite(loss):
                bad = (
                    f"NON-FINITE LOSS, skipping batch {b['key']} "
                    f"(attn {l_attn.item():.3g} warp {l_warp.item():.3g}"
                    f" conf {l_conf.item():.3g} l1 {l1.item():.3g}"
                    f" lpips {lp.item():.3g} gan {l_gan.item():.3g})"
                )
                break
            loss.backward()
            if train_disc:
                # 8-bit discretized fake, as upstream (both recon views)
                for xrv, imgv in (
                    (preds["recon_A"], b["img_A"]),
                    (preds["recon_B"], b["img_B"]),
                ):
                    fake = (xrv * 2 - 1).detach().clamp(-1.0, 1.0)
                    fakes.append(torch.round((fake + 1.0) * 127.5) / 127.5 - 1.0)
                    reals.append(imgv * 2 - 1)

        if bad is None:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            if not torch.isfinite(gnorm):
                bad = f"NON-FINITE GRAD NORM, skipping batch {b['key']}"
        if bad is not None:
            n_skipped += 1
            print(f"[{cfg.arm}] step {step} {bad} [{n_skipped} skipped total]",
                  flush=True)
            opt.zero_grad(set_to_none=True)
            sched.step()
            if train_disc:
                disc_sched.step()
            continue

        opt.step()
        sched.step()

        d_loss = None
        if train_disc:
            disc.train()
            disc_opt.zero_grad(set_to_none=True)
            d_loss = 0.0
            for fake, real in zip(fakes, reals):
                d = hinge_d_loss(
                    disc(disc_aug.aug(real)), disc(disc_aug.aug(fake))
                ) / len(fakes)
                d.backward()
                d_loss += d.item()
            disc_opt.step()
            disc_sched.step()
            disc.eval()

        if step % cfg.log.every == 0:
            msg = (
                f"[{cfg.arm}] step {step} attn {l_attn.item():.3f}"
                f" warp {l_warp.item():.4f} conf {l_conf.item():.4f}"
                f" l1 {l1.item():.4f} lpips {lp.item():.4f}"
            )
            if use_gan:
                msg += f" gan {l_gan.item():.4f} lam {lam.item():.3f}"
            if d_loss is not None:
                msg += f" d {d_loss:.4f}"
            print(msg, flush=True)

        if (step + 1) % cfg.log.ckpt_every == 0 or step + 1 == steps:
            state = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": step + 1,
                "arm": cfg.arm,
                "lam_rec": lam_rec,
            }
            if disc is not None:
                state.update({
                    "disc": disc.state_dict(),
                    "disc_opt": disc_opt.state_dict(),
                    "disc_sched": disc_sched.state_dict(),
                })
            torch.save(state, ckpt_path)
        if (step + 1) % cfg.log.eval_every == 0 and not flags.skip_head:
            m = evaluate(model, eval_ds, cfg.size, device, cfg.log.eval_pairs)
            print(
                f"[{cfg.arm}] step {step + 1} eval {json.dumps(m.get('all', {}))}",
                flush=True,
            )

    if flags.skip_head:
        metrics = {"recon_only": True, "note": "no matching pipeline run"}
    else:
        metrics = evaluate(model, eval_ds, cfg.size, device)
        print(f"[{cfg.arm}] final eval: {json.dumps(metrics, indent=1)}", flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if not flags.skip_head:
        save_panels(model, eval_ds, out_dir, cfg.size, device)
    if flags.with_decoder:
        save_recon_grid(model, eval_ds, out_dir, device)

    write_run_json(
        out_dir,
        {
            **OmegaConf.to_container(cfg, resolve=True),
            "lam_rec_effective": lam_rec,
            "matcher_lr_effective": m_lr,
            "gan_active": bool(disc is not None),
            "disc_start": disc_start if disc is not None else None,
            "gan_start": gan_start if disc is not None else None,
            "n_train_pairs": len(train_ds),
            "n_eval_pairs": len(eval_ds),
            "n_skipped_steps": n_skipped,
            "matcher": (
                "desc baseline: linear 2048->1024 in place of mv_vit, "
                "amp off, frozen precomputed desc, no refiners"
                if flags.desc_only
                else f"romav2 Matcher.Cfg defaults, {cfg.model.init} init, amp"
                " off, frozen precomputed desc, no refiners"
            ),
            "decoder": (
                f"RAEDecoder-{cfg.model.decoder_size} on mv_A+mv_B"
                if flags.with_decoder else None
            ),
            "params_M": n_params,
        },
    )
    return metrics
