"""R2 pre-flight: validate the hand-rolled CO3D camera/warp pipeline.

Two checks (docs/R2.md), both must pass before any R2 training:
1. pointcloud: project each sequence's pointcloud.ply into sample frames —
   points must land on the object. Tests K + world->cam conversion without
   involving depth.
2. photometric: warp image B into A's frame via the dense GT warp, compare
   inside the covisibility mask (masked PSNR + covis fraction). Tests the
   full depth/unproject/reproject/consistency path end to end.

Outputs results/r2_warp_check/{pointcloud,warp}_<seq>_<i>_<j>.png + a
summary line per pair. Uses the R1 hydrant-full split's sampling so the
checked pairs look like the training distribution (near + far).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from plyfile import PlyData  # noqa: E402

from src.co3d_geom import CO3D_ROOT, compute_warp, load_frame_index, meta_for  # noqa: E402
from src.data import list_sequences  # noqa: E402
from src.paths import RESULTS_DIR  # noqa: E402
from src.romav2_utils import img_to_tensor01  # noqa: E402
from src.viz.style import pyplot  # noqa: E402

plt = pyplot(report_style=False)
SIZE = 320
OUT = RESULTS_DIR / "r2_warp_check"


def load_img(path):
    return img_to_tensor01(Image.open(path), SIZE)[0].permute(1, 2, 0)  # (S,S,3)


def check_pointcloud(index, seq_name, frames, ax_row):
    ply = PlyData.read(str(CO3D_ROOT / "hydrant" / seq_name / "pointcloud.ply"))
    v = ply["vertex"]
    pts = torch.from_numpy(
        np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    )
    keep = torch.randperm(len(pts))[:4000]
    pts = pts[keep]
    from src.co3d_geom import project_points

    for ax, fi in zip(ax_row, [0, len(frames) // 2]):
        meta = meta_for(index, frames[fi])
        u, v_, z = project_points(meta, pts, SIZE)
        ax.imshow(load_img(frames[fi]).numpy())
        ok = z > 0
        ax.scatter(u[ok], v_[ok], s=0.3, c="lime", alpha=0.5)
        ax.set_xlim(0, SIZE), ax.set_ylim(SIZE, 0)
        ax.set_title(f"{seq_name[:12]} f{fi}", fontsize=7)
        ax.axis("off")


def check_pair(index, path_A, path_B, tag):
    warp, covis = compute_warp(index, path_A, path_B, SIZE)
    img_A, img_B = load_img(path_A), load_img(path_B)
    warped = torch.nn.functional.grid_sample(
        img_B.permute(2, 0, 1)[None],
        warp[None],
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    m = covis[..., None].float()
    covis_frac = covis.float().mean().item()
    if covis.any():
        mse = ((warped - img_A) ** 2 * m).sum() / (m.sum() * 3)
        psnr = -10 * torch.log10(mse).item()
    else:
        psnr = float("nan")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    panels = [
        (img_A, "A (target frame)"),
        (img_B, "B (source)"),
        (warped * m, f"B warped to A, covis {covis_frac:.0%}"),
        (0.5 * img_A + 0.5 * warped * m, f"blend, masked PSNR {psnr:.1f} dB"),
    ]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im.numpy().clip(0, 1))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"warp_{tag}.png", dpi=110)
    plt.close(fig)
    print(f"{tag}: covis {covis_frac:.3f} masked-PSNR {psnr:.2f} dB", flush=True)
    return covis_frac, psnr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = load_frame_index("hydrant")
    seqs = list_sequences("hydrant")[:3]

    fig, axes = plt.subplots(len(seqs), 2, figsize=(6, 3 * len(seqs)))
    for row, (seq_name, frames) in zip(np.atleast_2d(axes), seqs):
        check_pointcloud(index, seq_name, frames, row)
    fig.tight_layout()
    fig.savefig(OUT / "pointcloud.png", dpi=110)
    plt.close(fig)
    print("pointcloud overlays written", flush=True)

    results = []
    for seq_name, frames in seqs:
        n = len(frames)
        i = n // 4
        for j, kind in [(i + 2, "near"), ((i + n // 2) % n, "far")]:
            tag = f"{seq_name}_{i}_{j}_{kind}"
            results.append(check_pair(index, frames[i], frames[j], tag))
    fracs, psnrs = zip(*results)
    print(
        f"summary: covis mean {np.mean(fracs):.3f}, "
        f"masked-PSNR mean {np.nanmean(psnrs):.2f} dB",
        flush=True,
    )


if __name__ == "__main__":
    main()
