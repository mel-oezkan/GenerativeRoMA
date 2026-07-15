"""Figures + stats for the CO3D depth-map issues report (docs/co3d_depth_issues.md).

Documents the three depth problems hit while building R2's warp supervision
(see r2_warp_check.py for the validation runs that surfaced them):
  1. test_* frames ship all-zero depth (CO3D withholds test GT)
  2. MVS depth is noisy: holes / invalid regions, worst on far-field ground
  3. depth-consistency covisibility "sees through" objects at loose relative
     thresholds (scene depth ~15 units vs ~1-unit objects)
plus the non-problem lookalike: far-pair photometric residual is shading/
exposure, not bad geometry.

Outputs results/r2_depth_report/fig*.png + stats.json (numbers quoted in the
doc/report). CPU-only.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.co3d_geom import (
    CO3D_ROOT,
    compute_warp,
    load_depth_320,
    load_frame_index,
    meta_for,
)
from src.data import list_sequences
from src.romav2_utils import img_to_tensor01

SIZE = 320
OUT = Path(__file__).resolve().parent.parent / "results/r2_depth_report"
NEAR_C, FAR_C = "#2a78d6", "#eb6834"
STATS = {}


def load_img(path):
    return img_to_tensor01(Image.open(path), SIZE)[0].permute(1, 2, 0).numpy()


def img_path(index, seq, fname):
    return CO3D_ROOT / "hydrant" / seq / "images" / fname


def depth_panel(ax_img, ax_d, ax_m, index, seq, fname, title):
    p = img_path(index, seq, fname)
    meta = meta_for(index, p)
    d, valid = load_depth_320(meta, SIZE)
    vf = valid.float().mean().item()
    ax_img.imshow(load_img(p))
    ax_img.set_title(f"{title}\n{seq[:14]}/{fname[:13]}", fontsize=8)
    dm = d.clone()
    dm[~valid] = float("nan")
    im = ax_d.imshow(dm.numpy(), cmap="viridis")
    ax_d.set_title(f"depth ({meta['frame_type']})", fontsize=8)
    ax_m.imshow(valid.numpy(), cmap="gray", vmin=0, vmax=1)
    ax_m.set_title(f"valid: {vf:.1%}", fontsize=8)
    for ax in (ax_img, ax_d, ax_m):
        ax.axis("off")
    return vf


def fig1_test_frames(index):
    """Problem 1: test_* frames have all-zero depth."""
    # frame_type is homogeneous per sequence (no seq mixes train_* and
    # test_*), so contrast a train frame and a test frame from different seqs
    tr = next((s, f) for (s, f), m in index.items() if "train" in m["frame_type"])
    te = next((s, f) for (s, f), m in index.items() if "test" in m["frame_type"])
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
    vf_tr = depth_panel(*axes[0], index, *tr, "train frame (depth present)")
    vf_te = depth_panel(*axes[1], index, *te, "test frame (depth withheld)")
    seq = te[0]
    fig.tight_layout()
    fig.savefig(OUT / "fig1_test_frames.png", dpi=110)
    plt.close(fig)
    types = {}
    for m in index.values():
        types[m["frame_type"]] = types.get(m["frame_type"], 0) + 1
    STATS["frame_types"] = types
    STATS["test_frame_frac"] = sum(v for k, v in types.items() if "test" in k) / len(index)
    STATS["fig1"] = {"seq": seq, "valid_train": vf_tr, "valid_test": vf_te}


def fig2_mvs_noise(index):
    """Problem 2: MVS holes/noise — best/typical/worst valid fraction."""
    rng = np.random.default_rng(0)
    keys = [k for k, m in index.items() if "train" in m["frame_type"]]
    sample = [keys[i] for i in rng.choice(len(keys), 120, replace=False)]
    fracs = []
    for seq, fname in sample:
        _, valid = load_depth_320(index[(seq, fname)], SIZE)
        fracs.append((valid.float().mean().item(), seq, fname))
    fracs.sort()
    STATS["valid_frac_sample"] = {
        "n": len(fracs),
        "min": fracs[0][0], "p10": fracs[len(fracs) // 10][0],
        "median": fracs[len(fracs) // 2][0], "max": fracs[-1][0],
    }
    picks = [(fracs[-1], "best of sample"),
             (fracs[len(fracs) // 2], "median"),
             (fracs[0], "worst of sample")]
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 9.9))
    for row, ((vf, seq, fname), label) in zip(axes, picks):
        depth_panel(*row, index, seq, fname, label)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_mvs_noise.png", dpi=110)
    plt.close(fig)


def fig3_seethrough(index, path_A, path_B):
    """Problem 3: loose rel. threshold marks back-side surfaces covisible."""
    warp, covis_loose = compute_warp(index, path_A, path_B, SIZE, rel_thresh=0.10)
    _, covis_tight = compute_warp(index, path_A, path_B, SIZE, rel_thresh=0.01)
    false_c = covis_loose & ~covis_tight
    A, B = load_img(path_A), load_img(path_B)
    warped = F.grid_sample(
        torch.from_numpy(B).permute(2, 0, 1)[None].float(), warp[None],
        mode="bilinear", align_corners=False,
    )[0].permute(1, 2, 0).numpy()
    err = np.abs(warped - A).mean(-1)

    overlay = A.copy() * 0.55
    overlay[covis_tight.numpy()] += np.array([0, 0.45, 0])
    overlay[false_c.numpy()] += np.array([0.45, 0, 0])

    fig, axes = plt.subplots(2, 3, figsize=(9.9, 6.8))
    panels = [
        (A, "A (target view)"),
        (B, "B (source, ~opposite side)"),
        (overlay.clip(0, 1),
         "covis: green = kept (0.01)\nred = false at loose 0.10"),
        (warped * covis_loose.numpy()[..., None],
         "B warped to A, thresh 0.10\n(back-side texture in red zones)"),
        (warped * covis_tight.numpy()[..., None], "B warped to A, thresh 0.01"),
        (np.where(covis_loose.numpy(), err, np.nan), None),
    ]
    for ax, (im, t) in zip(axes.flat, panels):
        if t is None:
            m = ax.imshow(im, cmap="magma", vmin=0, vmax=0.6)
            ax.set_title("photometric error |warped-A|\n(thresh 0.10)", fontsize=8)
            plt.colorbar(m, ax=ax, fraction=0.045)
        else:
            ax.imshow(np.asarray(im).clip(0, 1))
            ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_seethrough.png", dpi=110)
    plt.close(fig)

    n_loose = covis_loose.sum().item()
    STATS["fig3"] = {
        "pair": f"{Path(path_A).parts[-3]} {Path(path_A).stem}->{Path(path_B).stem}",
        "covis_loose": covis_loose.float().mean().item(),
        "covis_tight": covis_tight.float().mean().item(),
        "false_frac_of_loose": false_c.sum().item() / max(n_loose, 1),
        "err_false": float(np.nanmean(np.where(false_c.numpy(), err, np.nan))),
        "err_true": float(np.nanmean(np.where(covis_tight.numpy(), err, np.nan))),
    }


def masked_psnr(index, path_A, path_B, th):
    warp, covis = compute_warp(index, path_A, path_B, SIZE, rel_thresh=th)
    A = torch.from_numpy(load_img(path_A)).float()
    B = torch.from_numpy(load_img(path_B)).float()
    w = F.grid_sample(B.permute(2, 0, 1)[None], warp[None], mode="bilinear",
                      align_corners=False)[0].permute(1, 2, 0)
    m = covis[..., None].float()
    if not covis.any():
        return covis.float().mean().item(), float("nan")
    mse = ((w - A) ** 2 * m).sum() / (m.sum() * 3)
    return covis.float().mean().item(), (-10 * torch.log10(mse)).item()


def fig4_threshold_sweep(index, pairs):
    ths = [0.005, 0.01, 0.02, 0.05, 0.10]
    curves = {"near": {"covis": [], "psnr": []}, "far": {"covis": [], "psnr": []}}
    for th in ths:
        acc = {"near": [], "far": []}
        for kind, pA, pB in pairs:
            acc[kind].append(masked_psnr(index, pA, pB, th))
        for kind in acc:
            curves[kind]["covis"].append(float(np.mean([a[0] for a in acc[kind]])))
            curves[kind]["psnr"].append(float(np.nanmean([a[1] for a in acc[kind]])))
    STATS["sweep"] = {"thresholds": ths, **curves}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for ax, metric, ylab in [(ax1, "covis", "covisible fraction of pixels"),
                             (ax2, "psnr", "masked photometric PSNR (dB)")]:
        for kind, c in [("near", NEAR_C), ("far", FAR_C)]:
            ax.plot(ths, curves[kind][metric], color=c, lw=2, marker="o", ms=6,
                    label=f"{kind} pairs")
            ax.annotate(f"{kind}", (ths[-1], curves[kind][metric][-1]),
                        textcoords="offset points", xytext=(8, -3),
                        fontsize=9, color="#0b0b0b")
        ax.axvline(0.01, color="#52514e", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(ths)
        ax.set_xticklabels([str(t) for t in ths], fontsize=8)
        ax.set_xlabel("relative depth-consistency threshold", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.margins(x=0.12)
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title("looser threshold inflates far-pair covisibility", fontsize=9)
    ax2.set_title("…with pixels that don't match photometrically", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_threshold.png", dpi=130)
    plt.close(fig)


def fig5_shading(index, path_A, path_B):
    """Non-problem: far-pair residual at tight threshold is shading, not
    geometry — the blend is structurally aligned."""
    warp, covis = compute_warp(index, path_A, path_B, SIZE, rel_thresh=0.01)
    A, B = load_img(path_A), load_img(path_B)
    warped = F.grid_sample(
        torch.from_numpy(B).permute(2, 0, 1)[None].float(), warp[None],
        mode="bilinear", align_corners=False,
    )[0].permute(1, 2, 0).numpy()
    m = covis.numpy()[..., None]
    fig, axes = plt.subplots(1, 3, figsize=(9.9, 3.8))
    for ax, im, t in zip(
        axes,
        [A, (warped * m), (0.5 * A + 0.5 * warped * m)],
        ["A", "B warped to A (thresh 0.01)",
         "blend: aligned structure,\ndifferent shading/exposure"],
    ):
        ax.imshow(np.asarray(im).clip(0, 1))
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig5_shading.png", dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = load_frame_index("hydrant")
    seqs = list_sequences("hydrant")[:3]
    pairs = []
    for seq, frames in seqs:
        n = len(frames)
        i = n // 4
        pairs.append(("near", frames[i], frames[i + 2]))
        pairs.append(("far", frames[i], frames[(i + n // 2) % n]))

    fig1_test_frames(index)
    print("fig1 done", flush=True)
    fig2_mvs_noise(index)
    print("fig2 done", flush=True)
    far = [(a, b) for k, a, b in pairs if k == "far"][0]
    fig3_seethrough(index, *far)
    print("fig3 done", flush=True)
    fig4_threshold_sweep(index, pairs)
    print("fig4 done", flush=True)
    fig5_shading(index, *far)
    print("fig5 done", flush=True)
    (OUT / "stats.json").write_text(json.dumps(STATS, indent=2))
    print(json.dumps(STATS, indent=1))


if __name__ == "__main__":
    main()
