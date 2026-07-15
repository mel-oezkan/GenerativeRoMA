"""CO3D geometry for R2: cameras, depths, dense GT warps + covisibility.

CO3D `frame_annotations.jgz` stores PyTorch3D-convention cameras
(row-vector world->cam: X_cam = X @ R + T; screen axes x-left/y-up; NDC
intrinsics, hydrant is all "ndc_isotropic"). We convert once to OpenCV
convention (column-vector, x-right/y-down) + pixel intrinsics:

    R_cv = diag(-1,-1,1) @ R^T          t_cv = diag(-1,-1,1) @ T
    s = min(W, H) / 2   (ndc_isotropic; per-axis W/2, H/2 for
                         ndc_norm_image_bounds)
    fx = f_ndc_x * s    cx = W/2 - pp_ndc_x * s   (same for y)

and then adjust K for the R1 image transform (Resize(short->size) +
CenterCrop(size), torchvision semantics: truncating long-side resize,
rounded crop offsets) so warps live in the same 320x320 frame as the
cached features. Continuous pixel coords put pixel i's center at i + 0.5;
normalized coords follow RoMaV2's grid ([-1 + 1/n, 1 - 1/n] centers, i.e.
align_corners=False, (x, y) order).

Depth PNGs are uint16 buffers reinterpreted as float16 (official CO3D
loader) times `scale_adjustment`; test_* frames ship all-zero depth, which
ends up as covisibility 0 -> callers must guard on `covis.sum()`.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CO3D_ROOT = Path("/visinf/projects_students/dlcv2025_groupZ/co3d_full")
META_CACHE_DIR = Path("/visinf/projects_students/dlcv2025_groupZ/romav2_feats/r2_meta")

_FLIP = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)


def _frame_meta(e):
    vp = e["viewpoint"]
    h, w = e["image"]["size"]
    R = np.asarray(vp["R"], dtype=np.float32)
    T = np.asarray(vp["T"], dtype=np.float32)
    f = np.asarray(vp["focal_length"], dtype=np.float32)
    pp = np.asarray(vp["principal_point"], dtype=np.float32)
    if vp["intrinsics_format"] == "ndc_isotropic":
        rescale = np.array([min(w, h) / 2] * 2, dtype=np.float32)
    elif vp["intrinsics_format"] == "ndc_norm_image_bounds":
        rescale = np.array([w / 2, h / 2], dtype=np.float32)
    else:
        raise ValueError(vp["intrinsics_format"])
    fx, fy = f * rescale
    cx, cy = np.array([w / 2, h / 2], dtype=np.float32) - pp * rescale
    return {
        "R_cv": _FLIP @ R.T,
        "t_cv": _FLIP @ T,
        "K": np.array([fx, fy, cx, cy], dtype=np.float32),  # original pixels
        "hw": (h, w),
        "depth_path": e["depth"]["path"],
        "depth_scale": float(e["depth"]["scale_adjustment"]),
        "depth_mask_path": e["depth"]["mask_path"],
        "frame_type": e["meta"]["frame_type"],
    }


def load_frame_index(category="hydrant"):
    """(sequence_name, image_filename) -> meta dict; parsed once, cached."""
    cache = META_CACHE_DIR / f"{category}_frames.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    with gzip.open(CO3D_ROOT / category / "frame_annotations.jgz", "rt") as f:
        data = json.load(f)
    index = {
        (e["sequence_name"], Path(e["image"]["path"]).name): _frame_meta(e)
        for e in data
    }
    META_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(index, cache)
    return index


def load_seq_quality(category="hydrant"):
    """sequence_name -> viewpoint_quality_score (CO3D's SfM pose-quality
    score; NaN when absent). Bad poses make the warp GT *systematically*
    wrong — the depth-consistency check reuses the same poses, so it cannot
    catch them (docs/co3d_depth_issues.md); filter sequences instead."""
    with gzip.open(CO3D_ROOT / category / "sequence_annotations.jgz", "rt") as f:
        data = json.load(f)
    return {
        s["sequence_name"]: (
            float("nan") if s["viewpoint_quality_score"] is None
            else s["viewpoint_quality_score"]
        )
        for s in data
    }


def meta_for(index, frame_path):
    """Look up by image path (…/<seq>/images/<frame>.jpg)."""
    p = Path(frame_path)
    return index[(p.parts[-3], p.name)]


def resized_cropped_K(meta, size):
    """K after Resize(short_side=size) + CenterCrop(size), torchvision
    semantics (matches src.romav2_utils.img_to_tensor01)."""
    h, w = meta["hw"]
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_w, new_h = int(size * w / h), size
    sx, sy = new_w / w, new_h / h
    top = int(round((new_h - size) / 2.0))
    left = int(round((new_w - size) / 2.0))
    fx, fy, cx, cy = meta["K"]
    return (
        np.array([fx * sx, fy * sy, cx * sx - left, cy * sy - top], np.float32),
        (new_h, new_w),
        (top, left),
    )


def load_depth_320(meta, size):
    """Depth + validity in the transformed size x size frame.

    Nearest resize (interpolating depth across silhouettes creates fake
    geometry), then the same center crop as the image transform.
    """
    with Image.open(CO3D_ROOT / meta["depth_path"]) as dpil:
        d = (
            np.frombuffer(np.array(dpil, np.uint16).tobytes(), np.float16)
            .astype(np.float32)
            .reshape(dpil.size[1], dpil.size[0])
        ) * meta["depth_scale"]
    with Image.open(CO3D_ROOT / meta["depth_mask_path"]) as mpil:
        m = np.array(mpil)
    if m.ndim == 3:
        m = m[..., 0]
    _, (new_h, new_w), (top, left) = resized_cropped_K(meta, size)
    d_t = torch.from_numpy(d.copy())[None, None]
    m_t = torch.from_numpy((m > 0).astype(np.float32))[None, None]
    d_r = torch.nn.functional.interpolate(d_t, (new_h, new_w), mode="nearest-exact")
    m_r = torch.nn.functional.interpolate(m_t, (new_h, new_w), mode="nearest-exact")
    d_c = d_r[0, 0, top : top + size, left : left + size]
    m_c = m_r[0, 0, top : top + size, left : left + size]
    # depth files are fp16, so overflowed pixels are stored as inf; a single
    # inf reaching the warp poisons downstream losses (0 * inf = nan)
    valid = (d_c > 0) & (m_c > 0) & torch.isfinite(d_c)
    d_c = torch.nan_to_num(d_c, nan=0.0, posinf=0.0, neginf=0.0)
    return d_c, valid


def compute_warp(index, path_A, path_B, size=320, rel_thresh=0.01):
    """Dense GT warp A->B and covisibility in the size x size frame.

    Returns (warp[size,size,2] normalized B-coords, covis[size,size] bool).
    Covis = valid depth in A, projects in front of and inside B, and B's
    depth at the target agrees within rel_thresh. 0.01 validated 2026-07-14
    (r2_warp_check): CO3D scene depth is ~15 units vs ~1-unit objects, so
    looser thresholds accept back-side surfaces through the object; 0.01
    removes those without reducing near-pair covis (0.127 -> 0.126).
    All-zero depth (test frames) yields covis.sum() == 0.
    """
    mA, mB = meta_for(index, path_A), meta_for(index, path_B)
    d_A, valid_A = load_depth_320(mA, size)
    d_B, valid_B = load_depth_320(mB, size)
    return _warp_one_dir(mA, d_A, valid_A, mB, d_B, valid_B, size, rel_thresh)


def compute_warp_pair(index, path_A, path_B, size=320, rel_thresh=0.01):
    """Both directions from one set of depth loads:
    (warp_AB, covis_AB, warp_BA, covis_BA)."""
    mA, mB = meta_for(index, path_A), meta_for(index, path_B)
    d_A, valid_A = load_depth_320(mA, size)
    d_B, valid_B = load_depth_320(mB, size)
    return _warp_one_dir(mA, d_A, valid_A, mB, d_B, valid_B, size, rel_thresh) + \
        _warp_one_dir(mB, d_B, valid_B, mA, d_A, valid_A, size, rel_thresh)


def _warp_one_dir(mA, d_A, valid_A, mB, d_B, valid_B, size, rel_thresh):
    K_A, _, _ = resized_cropped_K(mA, size)
    K_B, _, _ = resized_cropped_K(mB, size)

    ys, xs = torch.meshgrid(
        torch.arange(size) + 0.5, torch.arange(size) + 0.5, indexing="ij"
    )
    fx, fy, cx, cy = K_A
    x_cam = torch.stack(
        [(xs - cx) / fx * d_A, (ys - cy) / fy * d_A, d_A], dim=-1
    )  # (S,S,3) in A's camera
    R_A = torch.from_numpy(mA["R_cv"])
    t_A = torch.from_numpy(mA["t_cv"])
    R_B = torch.from_numpy(mB["R_cv"])
    t_B = torch.from_numpy(mB["t_cv"])
    X = (x_cam - t_A) @ R_A  # world: R_A^T (x - t_A), row-vector form
    x_b = X @ R_B.T + t_B
    z = x_b[..., 2]
    fx_b, fy_b, cx_b, cy_b = K_B
    eps = 1e-6
    u = fx_b * x_b[..., 0] / z.clamp(min=eps) + cx_b
    v = fy_b * x_b[..., 1] / z.clamp(min=eps) + cy_b
    warp = torch.stack([2 * u / size - 1, 2 * v / size - 1], dim=-1)

    in_bounds = (warp.abs() < 1 - 1e-4).all(dim=-1) & (z > eps)
    # sample B's depth (+ validity) at the warp target
    grid = warp[None]
    d_B_s = torch.nn.functional.grid_sample(
        d_B[None, None], grid, mode="bilinear", align_corners=False
    )[0, 0]
    v_B_s = torch.nn.functional.grid_sample(
        valid_B[None, None].float(), grid, mode="nearest", align_corners=False
    )[0, 0]
    consistent = (z - d_B_s).abs() / d_B_s.clamp(min=eps) < rel_thresh
    covis = valid_A & in_bounds & (v_B_s > 0) & (d_B_s > 0) & consistent
    return warp, covis


def project_points(meta, pts_world, size=320):
    """World points -> (u, v, z) in the transformed size x size pixel frame
    (pointcloud sanity check for the convention conversion)."""
    K, _, _ = resized_cropped_K(meta, size)
    R = torch.from_numpy(meta["R_cv"])
    t = torch.from_numpy(meta["t_cv"])
    x = pts_world @ R.T + t
    z = x[..., 2]
    fx, fy, cx, cy = K
    u = fx * x[..., 0] / z.clamp(min=1e-6) + cx
    v = fy * x[..., 1] / z.clamp(min=1e-6) + cy
    return u, v, z
