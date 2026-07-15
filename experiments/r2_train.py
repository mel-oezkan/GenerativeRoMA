"""R2 v0: RoMaV2 matcher trained FROM SCRATCH on CO3D hydrants with a joint
matching + reconstruction objective (docs/R2.md).

Stock Matcher architecture (mv_vit vit_base, DPT head -> warp+conf @ /4),
random init, frozen precomputed DINOv3 desc inputs (r2_precompute_desc.py),
320 px, no refiners. Losses per direction (bidirectional):
  - attn CE @ /16: cross-entropy of the global correlation logits against
    the GT match cell, soft-binned bilinearly over 4 neighbor cells
    (RoMa-style regression-by-classification), covisible tokens only
  - warp Charbonnier @ /4 + BCE(confidence, covisibility)
  - recon: RAEDecoder-B on the mv tokens of view A -> image A,
    L1 + 1.0*LPIPS(VGG); no GAN at training time (stability); the
    *measurement* remains the post-hoc R1 probe protocol

Arms (--arm): match (lam_rec=0) | joint (matching + lam_rec*recon) |
recon (recon only; matching-loss weights zeroed).

GT warps come from CO3D depth+pose on the fly (src/co3d_geom.py, validated
by r2_warp_check.py). Pairs = R1 hydrant-full split; pairs touching test_*
frames (all-zero depth) are dropped from the matching supervision. Eval:
EPE/PCK of the /4 warp + coarse argmax EPE @ /16 on held-out pairs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.co3d_geom import (
    META_CACHE_DIR,
    compute_warp_pair,
    load_frame_index,
    load_seq_quality,
    meta_for,
)
from src.decoders import RAEDecoder
from src.romav2_utils import ROMAV2_SRC, img_to_tensor01
from src.utils import warmup_cosine_lambda, write_run_json

if str(ROMAV2_SRC) not in sys.path:
    sys.path.insert(0, str(ROMAV2_SRC))

SIZE = 320
G16, G4 = SIZE // 16, SIZE // 4
DESC_CH, DESC_LAYER_CH = 2048, 1024
STEPS = 6000
BATCH = 4  # 6 OOMs on 11 GB in the joint arm (fp32 DPT head + LPIPS peak)
LR = 2e-4
MIN_LR_RATIO = 0.1
WARMUP = 300
WD_MATCHER = 0.05
GRAD_CLIP = 1.0
W_ATTN, W_WARP, W_CONF = 1.0, 5.0, 1.0
LPIPS_W = 1.0
CKPT_EVERY = 500
SEED = 0
DEVICE = "cuda"
DESC_CACHE = Path("/visinf/projects_students/dlcv2025_groupZ/romav2_feats/r2_desc_320")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results/r2_v0"
# dataset filters (docs/co3d_depth_issues.md): pose-quality floor for train
# sequences and a pair covisibility floor (tokens on the /16 grid, best
# direction) from the r2_pair_covis.py census
VQ_MIN = 0.5
COVIS_FLOOR = 8
PAIR_COVIS_CACHE = META_CACHE_DIR / "pair_covis_hydrant.json"


# ---------------------------------------------------------------- data


def frame_key(path):
    p = Path(path)
    return f"hydrant_{p.parts[-3]}_{p.stem}"


def has_depth(index, path):
    return "test" not in meta_for(index, path)["frame_type"]


def seq_of(pair):
    return Path(pair["anchor"]).parts[-3]


def build_train_pairs(index, verbose=True):
    """Filtered R2 pair set, shared by training and desc precompute so both
    see the same frames. Applied to *all* arms (identical data across arms):
      1. drop pairs touching test_* frames (zero depth, sometimes black RGB)
      2. drop pairs from sequences with viewpoint_quality_score < VQ_MIN or
         NaN (bad SfM poses = systematically wrong warp GT that the
         depth-consistency check cannot catch)
    Eval pairs are returned unfiltered (identical to R1 for comparability);
    the covis floor is applied separately in train() from the census cache.
    """
    from experiments.r1_recon_probe import get_splits

    train_pairs, eval_pairs = get_splits("hydrant-full")
    n0 = len(train_pairs)
    train_pairs = [p for p in train_pairs
                   if has_depth(index, p["anchor"]) and has_depth(index, p["other"])]
    n1 = len(train_pairs)
    vq = load_seq_quality("hydrant")
    train_pairs = [p for p in train_pairs
                   if vq.get(seq_of(p), float("nan")) >= VQ_MIN]  # NaN fails
    if verbose:
        print(f"train pairs: {n0} -> {n1} (test-frame filter) -> "
              f"{len(train_pairs)} (seq pose quality >= {VQ_MIN})", flush=True)
        ev = sorted({seq_of(p) for p in eval_pairs})
        print("eval seq quality:",
              {s: round(vq.get(s, float("nan")), 2) for s in ev}, flush=True)
    return train_pairs, eval_pairs


class R2PairDataset(torch.utils.data.Dataset):
    """desc (cached) + images + GT warps/covis at /16 and /4, both dirs.

    GT grids subsample the 320px warp at patch-center pixels (offset 8
    stride 16, offset 2 stride 4; centers are 0.5 px off the continuous
    patch center — negligible at feature granularity).
    """

    def __init__(self, pairs, index):
        self.pairs = pairs
        self.index = index

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        out = {"key": p["key"]}
        for tag, path in [("A", p["anchor"]), ("B", p["other"])]:
            out[f"desc_{tag}"] = torch.load(
                DESC_CACHE / f"{frame_key(path)}.pt",
                map_location="cpu", weights_only=True,
            )["desc"].float()
            out[f"img_{tag}"] = img_to_tensor01(Image.open(path), SIZE)[0]
        wab, cab, wba, cba = compute_warp_pair(self.index, p["anchor"], p["other"], SIZE)
        for tag, w, c in [("ab", wab, cab), ("ba", wba, cba)]:
            out[f"warp16_{tag}"] = w[8::16, 8::16]
            out[f"covis16_{tag}"] = c[8::16, 8::16]
            out[f"warp4_{tag}"] = w[2::4, 2::4]
            out[f"covis4_{tag}"] = c[2::4, 2::4]
        return out


# ---------------------------------------------------------------- model


class R2Model(torch.nn.Module):
    """From-scratch Matcher + (optional) recon decoder on the mv tokens."""

    def __init__(self, with_decoder):
        super().__init__()
        from romav2.matcher import Matcher

        self.matcher = Matcher(Matcher.Cfg(enable_amp=False))
        self.matcher.head.enable_amp = False  # vendored dpt.py patch (R2)
        self._mv = []
        self.matcher.mv_vit.register_forward_hook(
            lambda m, inp, out: self._mv.append(out["x_norm_patchtokens"])
        )
        self.decoder = RAEDecoder(1024, 16, size="B") if with_decoder else None

    def forward(self, desc_A, desc_B, img_A, img_B):
        # cached desc (B,2048,20,20) -> the Matcher's two-layer f_list format
        def to_list(d):
            d = d.permute(0, 2, 3, 1)
            return [d[..., :DESC_LAYER_CH], d[..., DESC_LAYER_CH:]]

        self._mv.clear()
        preds = self.matcher(
            to_list(desc_A), to_list(desc_B), img_A=img_A, img_B=img_B,
            bidirectional=True,
        )
        B = desc_A.shape[0]
        mv = self._mv.pop().reshape(B, 2, G16, G16, -1)
        preds["mv_A"] = mv[:, 0]
        if self.decoder is not None:
            preds["recon_A"] = self.decoder(
                mv[:, 0].permute(0, 3, 1, 2).contiguous()
            )
        return preds


# ---------------------------------------------------------------- losses


def attn_ce_loss(logits, warp16, covis16):
    """CE of (B,Ha,Wa,Hb,Wb) logits vs bilinear soft GT cells; covis only."""
    B = logits.shape[0]
    logp = F.log_softmax(logits.reshape(B, G16 * G16, G16 * G16), dim=-1)
    # non-covis cells can carry non-finite warps (fp16 depth overflow);
    # they are masked below, but nan.long() indexes out of bounds first
    warp16 = torch.nan_to_num(warp16, nan=0.0, posinf=0.0, neginf=0.0)
    c = ((warp16 + 1) * G16 / 2 - 0.5).clamp(0, G16 - 1)  # (B,20,20,2) cell coords
    f0 = c.floor().clamp(max=G16 - 2)
    w1 = (c - f0).clamp(0, 1)
    x0, y0 = f0[..., 0].long(), f0[..., 1].long()
    wx1, wy1 = w1[..., 0], w1[..., 1]
    loss = 0.0
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        w = ((wx1 if dx else 1 - wx1) * (wy1 if dy else 1 - wy1)).reshape(B, -1)
        idx = ((y0 + dy) * G16 + (x0 + dx)).reshape(B, -1)
        loss = loss - w * logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    m = covis16.reshape(B, -1).float()
    return (loss * m).sum() / m.sum().clamp(min=1)


def warp_conf_loss(warp_pred, conf_pred, warp4, covis4):
    m = covis4.float()
    # zero (not multiply-mask) non-covis residuals: 0 * inf = nan
    warp4 = torch.nan_to_num(warp4, nan=0.0, posinf=0.0, neginf=0.0)
    charb = torch.sqrt(((warp_pred - warp4) ** 2).sum(-1) + 1e-6)
    l_warp = torch.where(covis4, charb, 0.0).sum() / m.sum().clamp(min=1)
    l_conf = F.binary_cross_entropy_with_logits(conf_pred[..., 0], m)
    return l_warp, l_conf


def matching_loss(preds, batch):
    l_attn = attn_ce_loss(preds["attn_AB_logits"], batch["warp16_ab"], batch["covis16_ab"]) \
        + attn_ce_loss(preds["attn_BA_logits"], batch["warp16_ba"], batch["covis16_ba"])
    l_w_ab, l_c_ab = warp_conf_loss(
        preds["warp_AB"], preds["confidence_AB"], batch["warp4_ab"], batch["covis4_ab"]
    )
    l_w_ba, l_c_ba = warp_conf_loss(
        preds["warp_BA"], preds["confidence_BA"], batch["warp4_ba"], batch["covis4_ba"]
    )
    return l_attn / 2, (l_w_ab + l_w_ba) / 2, (l_c_ab + l_c_ba) / 2


# ---------------------------------------------------------------- eval


@torch.no_grad()
def evaluate(model, eval_ds, max_pairs=None):
    """EPE/PCK of the /4 warp + coarse argmax EPE @ /16, AB direction,
    covisible pixels; near/far split by |i - j| of the pair key."""
    model.eval()
    rows = []
    n = len(eval_ds) if max_pairs is None else min(max_pairs, len(eval_ds))
    for i in range(n):
        b = eval_ds[i]
        if b["covis4_ab"].sum() < 16:
            continue
        batch = {
            k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v)
            for k, v in b.items()
        }
        preds = model(batch["desc_A"], batch["desc_B"], batch["img_A"], batch["img_B"])
        m4 = batch["covis4_ab"][0]
        epe4 = ((preds["warp_AB"][0] - batch["warp4_ab"][0]) / 2 * SIZE).norm(dim=-1)[m4]
        m16 = batch["covis16_ab"][0]
        am = preds["attn_AB_logits"][0].reshape(G16 * G16, G16 * G16).argmax(-1)
        cell = torch.stack([am % G16, am // G16], -1).reshape(G16, G16, 2).float()
        coarse = (cell + 0.5) * (2 / G16) - 1
        epe16 = ((coarse - batch["warp16_ab"][0]) / 2 * SIZE).norm(dim=-1)[m16]
        ij = b["key"].split("_")[-2:]
        rows.append({
            "near": abs(int(ij[0]) - int(ij[1])) <= 4,
            "epe": epe4.mean().item(),
            "pck1": (epe4 < 1).float().mean().item(),
            "pck3": (epe4 < 3).float().mean().item(),
            "pck5": (epe4 < 5).float().mean().item(),
            "epe_coarse": epe16.mean().item() if m16.any() else float("nan"),
        })
    model.train()
    out = {}
    for split in ["all", "near", "far"]:
        rs = [r for r in rows
              if split == "all" or r["near"] == (split == "near")]
        if rs:
            out[split] = {k: float(np.nanmean([r[k] for r in rs]))
                          for k in ["epe", "pck1", "pck3", "pck5", "epe_coarse"]}
            out[split]["n"] = len(rs)
    return out


def save_panels(model, eval_ds, out_dir, n=4):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.3 * n))
    with torch.no_grad():
        shown = 0
        for i in range(len(eval_ds)):
            if shown >= n:
                break
            b = eval_ds[i]
            if b["covis4_ab"].sum() < 16:
                continue
            batch = {k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v)
                     for k, v in b.items()}
            preds = model(batch["desc_A"], batch["desc_B"],
                          batch["img_A"], batch["img_B"])
            conf = preds["confidence_AB"][0, ..., 0].sigmoid()
            warped = F.grid_sample(
                batch["img_B"], preds["warp_AB"][0][None].clamp(-1, 1),
                mode="bilinear", align_corners=False,
            )[0]
            warped = F.interpolate(warped[None], (SIZE, SIZE), mode="bilinear")[0]
            conf_up = F.interpolate(conf[None, None], (SIZE, SIZE), mode="bilinear")[0, 0]
            panels = [
                (b["img_A"].permute(1, 2, 0), "A"),
                (b["img_B"].permute(1, 2, 0), "B"),
                ((warped * (conf_up > 0.5)).permute(1, 2, 0).cpu(), "B warped by pred (conf>.5)"),
                (conf_up.cpu(), "pred confidence"),
            ]
            for ax, (im, t) in zip(axes[shown], panels):
                ax.imshow(im.numpy().clip(0, 1) if im.dim() == 3 else im.numpy(),
                          cmap=None if im.dim() == 3 else "viridis")
                ax.set_title(f"{t} [{b['key'][:24]}]" if t == "A" else t, fontsize=7)
                ax.axis("off")
            shown += 1
    fig.tight_layout()
    fig.savefig(out_dir / "warp_panels.png", dpi=110)
    plt.close(fig)
    model.train()


def save_recon_grid(model, eval_ds, out_dir, n=6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    with torch.no_grad():
        for col in range(n):
            b = eval_ds[col * (len(eval_ds) // n)]
            batch = {k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v)
                     for k, v in b.items()}
            preds = model(batch["desc_A"], batch["desc_B"],
                          batch["img_A"], batch["img_B"])
            axes[0, col].imshow(b["img_A"].permute(1, 2, 0).numpy())
            axes[1, col].imshow(
                preds["recon_A"][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            )
            for r in range(2):
                axes[r, col].axis("off")
    axes[0, 0].set_title("target", loc="left")
    axes[1, 0].set_title("recon (train-time decoder)", loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "recon_grid.png", dpi=120)
    plt.close(fig)
    model.train()


# ---------------------------------------------------------------- train


def collate(items):
    out = {"key": [b["key"] for b in items]}
    for k in items[0]:
        if k != "key":
            out[k] = torch.stack([b[k] for b in items])
    return out


def train(arm, lam_rec, steps, batch_size, out_root=None):
    from taming.modules.losses.lpips import LPIPS as TamingLPIPS

    out_dir = (Path(out_root) if out_root else RESULTS_DIR) / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    index = load_frame_index("hydrant")

    train_pairs, eval_pairs = build_train_pairs(index)
    use_matching = arm != "recon"
    if PAIR_COVIS_CACHE.exists():
        cc = json.loads(PAIR_COVIS_CACHE.read_text())
        n = len(train_pairs)
        # missing keys default to keep; floor on the better direction
        train_pairs = [p for p in train_pairs
                       if max(cc.get(p["key"], [COVIS_FLOOR] * 2)) >= COVIS_FLOOR]
        print(f"covis floor {COVIS_FLOOR}/400 tokens: {n} -> {len(train_pairs)}",
              flush=True)
    else:
        print(f"warning: {PAIR_COVIS_CACHE} missing, covis floor not applied",
              flush=True)
    print(f"final: {len(train_pairs)} train / {len(eval_pairs)} eval pairs", flush=True)

    train_ds = R2PairDataset(train_pairs, index)
    eval_ds = R2PairDataset(eval_pairs, index)

    torch.manual_seed(SEED)
    with_decoder = arm != "match"
    model = R2Model(with_decoder).to(DEVICE).train()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"arm {arm}: {n_params:.1f}M params (decoder: {with_decoder})", flush=True)

    groups = [{"params": model.matcher.parameters(), "weight_decay": WD_MATCHER}]
    if model.decoder is not None:
        groups.append({"params": model.decoder.parameters(), "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine_lambda(WARMUP, steps, MIN_LR_RATIO)
    )
    lpips_fn = None
    if with_decoder:
        lpips_fn = TamingLPIPS().to(DEVICE).eval().requires_grad_(False)

    # pre-drawn index sequence -> exact resume without replaying data loads
    g = torch.Generator().manual_seed(SEED)
    idx_all = torch.randint(len(train_ds), (steps * batch_size,), generator=g)

    ckpt_path = out_dir / "ckpt.pt"
    start = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start = ckpt["step"]
        print(f"resumed at step {start}", flush=True)

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_sampler=idx_all[start * batch_size :].reshape(-1, batch_size).tolist(),
        num_workers=4,
        collate_fn=collate,
        pin_memory=True,
    )
    it = iter(loader)

    n_skipped = 0
    for step in range(start, steps):
        b = next(it)
        b = {k: v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v
             for k, v in b.items()}
        opt.zero_grad(set_to_none=True)
        preds = model(b["desc_A"], b["desc_B"], b["img_A"], b["img_B"])

        zero = torch.zeros((), device=DEVICE)
        l_attn = l_warp = l_conf = l1 = lp = zero
        if use_matching:
            l_attn, l_warp, l_conf = matching_loss(preds, b)
        if with_decoder:
            xr = preds["recon_A"]
            l1 = (xr - b["img_A"]).abs().mean()
            lp = lpips_fn(b["img_A"] * 2 - 1, xr * 2 - 1).mean()
        loss = (W_ATTN * l_attn + W_WARP * l_warp + W_CONF * l_conf
                + lam_rec * (l1 + LPIPS_W * lp))
        # NaN guard: a single bad batch (e.g. inf in fp16 depth GT) must not
        # poison the weights forever — skip the step, keep the schedule
        if not torch.isfinite(loss):
            n_skipped += 1
            print(f"[{arm}] step {step} NON-FINITE LOSS, skipping batch "
                  f"{b['key']} (attn {l_attn.item():.3g} warp {l_warp.item():.3g}"
                  f" conf {l_conf.item():.3g} l1 {l1.item():.3g} lpips {lp.item():.3g})"
                  f" [{n_skipped} skipped total]", flush=True)
            sched.step()
            continue
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not torch.isfinite(gnorm):
            n_skipped += 1
            print(f"[{arm}] step {step} NON-FINITE GRAD NORM, skipping batch "
                  f"{b['key']} [{n_skipped} skipped total]", flush=True)
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.step()
        sched.step()

        if step % 50 == 0:
            print(f"[{arm}] step {step} attn {l_attn.item():.3f} warp {l_warp.item():.4f}"
                  f" conf {l_conf.item():.4f} l1 {l1.item():.4f} lpips {lp.item():.4f}",
                  flush=True)
        if (step + 1) % CKPT_EVERY == 0 or step + 1 == steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1,
                        "arm": arm, "lam_rec": lam_rec}, ckpt_path)
        if (step + 1) % 1000 == 0:
            m = evaluate(model, eval_ds, max_pairs=40)
            print(f"[{arm}] step {step + 1} eval {json.dumps(m.get('all', {}))}",
                  flush=True)

    metrics = evaluate(model, eval_ds)
    print(f"[{arm}] final eval: {json.dumps(metrics, indent=1)}", flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_panels(model, eval_ds, out_dir)
    if with_decoder:
        save_recon_grid(model, eval_ds, out_dir)
    write_run_json(out_dir, {
        "arm": arm, "lam_rec": lam_rec, "steps": steps, "batch": batch_size,
        "size": SIZE, "lr": LR, "warmup": WARMUP, "wd_matcher": WD_MATCHER,
        "w_attn": W_ATTN, "w_warp": W_WARP, "w_conf": W_CONF,
        "lpips_w": LPIPS_W, "grad_clip": GRAD_CLIP, "seed": SEED,
        "n_train_pairs": len(train_pairs), "n_eval_pairs": len(eval_pairs),
        "n_skipped_steps": n_skipped,
        "matcher": "romav2 Matcher.Cfg defaults, random init, amp off,"
                   " frozen precomputed desc, no refiners",
        "decoder": "RAEDecoder-B on mv_A" if with_decoder else None,
        "params_M": n_params,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["match", "joint", "recon"])
    ap.add_argument("--lam-rec", type=float, default=1.0,
                    help="recon weight (joint/recon arms)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--out-dir", default=None,
                    help="results root override (smoke runs)")
    args = ap.parse_args()
    lam = 0.0 if args.arm == "match" else args.lam_rec
    train(args.arm, lam, args.steps, args.batch, args.out_dir)


if __name__ == "__main__":
    main()
