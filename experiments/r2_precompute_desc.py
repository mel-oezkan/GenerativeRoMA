"""R2: per-frame DINOv3 descriptor cache for hydrant-full pairs.

R1 cached desc per *pair* (view A only); the matcher needs desc for both
views and frames repeat across pairs, so R2 caches per frame:
    romav2_feats/r2_desc_320/hydrant_<seq>_<frame_stem>.pt
        {"desc": (2048, 20, 20) fp16, "path": <image path>}
2048 = the two Descriptor layers (1024 each) channel-concatenated in list
order; split back into [.. :1024], [1024: ..] for Matcher's f_list input.

Uses the full pretrained RoMaV2's descriptor (identical to R1's desc) —
the descriptor is frozen in training, so pretrained-ness is by design.
--shard i/n for multi-GPU (CUDA_VISIBLE_DEVICES selects the GPU).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from src.romav2_utils import img_to_tensor01, load_romav2_frozen

SIZE = 320
CACHE_DIR = Path("/visinf/projects_students/dlcv2025_groupZ/romav2_feats/r2_desc_320")
DEVICE = "cuda"


def frame_key(path):
    p = Path(path)
    return f"hydrant_{p.parts[-3]}_{p.stem}"


def unique_frames(pairs):
    frames = {}
    for p in pairs:
        for f in (p["anchor"], p["other"]):
            frames[frame_key(f)] = f
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="i/n slice for parallel GPUs")
    args = ap.parse_args()

    from experiments.r2_train import build_train_pairs
    from src.co3d_geom import load_frame_index

    # same filtered pair set as training (covis floor not applied here — a
    # frame dropped by it may survive in another pair; superset is fine)
    train_pairs, eval_pairs = build_train_pairs(load_frame_index("hydrant"))
    frames = sorted(unique_frames(train_pairs + eval_pairs).items())
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        frames = frames[i::n]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(k, f) for k, f in frames if not (CACHE_DIR / f"{k}.pt").exists()]
    print(f"{len(frames)} frames in shard, {len(todo)} to compute", flush=True)
    if not todo:
        return

    model = load_romav2_frozen()
    with torch.no_grad():
        for n_done, (key, path) in enumerate(todo):
            img = img_to_tensor01(Image.open(path), SIZE).to(DEVICE)
            f_list = model.f(img)  # tuple of (1, 20, 20, 1024)
            desc = torch.cat(f_list, dim=-1)[0].permute(2, 0, 1).contiguous()
            torch.save(
                {"desc": desc.half().cpu(), "path": str(path)},
                CACHE_DIR / f"{key}.pt",
            )
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(todo)}", flush=True)
    print("desc cache complete", flush=True)


if __name__ == "__main__":
    main()
