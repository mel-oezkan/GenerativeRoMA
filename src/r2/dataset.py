"""R2/R3 pair data: the filter pipeline and the training dataset.

The filters are applied identically in training, in the covisibility census
and in the desc precompute, so all three see the same frames — that is the
whole reason they live here instead of in the training script.
"""

import json
from pathlib import Path

import torch
from PIL import Image

from src.co3d_geom import compute_warp_pair, load_seq_quality, meta_for
from src.paths import DESC_CACHE_DIR
from src.romav2_utils import img_to_tensor01
from src.splits import categories_for, covis_cache_for, get_splits


def frame_key(path):
    p = Path(path)  # .../co3d_full/<category>/<seq>/images/<frame>.jpg
    return f"{p.parts[-4]}_{p.parts[-3]}_{p.stem}"


def has_depth(index, path):
    return "test" not in meta_for(index, path)["frame_type"]


def seq_of(pair):
    return Path(pair["anchor"]).parts[-3]


def build_train_pairs(index, split="hydrant-full", vq_min=0.5, verbose=True):
    """Filtered R2 pair set, shared by training and desc precompute so both
    see the same frames. Applied to *all* arms (identical data across arms):
      1. drop pairs touching test_* frames (zero depth, sometimes black RGB)
      2. drop pairs from sequences with viewpoint_quality_score < vq_min or
         NaN (bad SfM poses = systematically wrong warp GT that the
         depth-consistency check cannot catch)
    Eval pairs are returned unfiltered (identical to R1 for comparability);
    the covis floor is applied separately (train/precompute) from the
    census cache.
    """
    train_pairs, eval_pairs = get_splits(split)
    n0 = len(train_pairs)
    train_pairs = [
        p
        for p in train_pairs
        if has_depth(index, p["anchor"]) and has_depth(index, p["other"])
    ]
    n1 = len(train_pairs)
    vq = {}
    for cat in categories_for(split):
        vq.update(load_seq_quality(cat))
    train_pairs = [
        p for p in train_pairs if vq.get(seq_of(p), float("nan")) >= vq_min
    ]  # NaN fails
    if verbose:
        print(
            f"train pairs: {n0} -> {n1} (test-frame filter) -> "
            f"{len(train_pairs)} (seq pose quality >= {vq_min})",
            flush=True,
        )
        ev = sorted({seq_of(p) for p in eval_pairs})
        print(
            "eval seq quality:",
            {s: round(vq.get(s, float("nan")), 2) for s in ev},
            flush=True,
        )
    return train_pairs, eval_pairs


def apply_covis_floor(train_pairs, split, floor=8):
    """Drop train pairs whose better direction has < floor covisible tokens
    (r2_pair_covis.py census). Missing keys default to keep."""
    cache = covis_cache_for(split)
    if not cache.exists():
        print(f"warning: {cache} missing, covis floor not applied", flush=True)
        return train_pairs
    cc = json.loads(cache.read_text())
    kept = [p for p in train_pairs if max(cc.get(p["key"], [floor] * 2)) >= floor]
    print(
        f"covis floor {floor}/400 tokens: {len(train_pairs)} -> {len(kept)}", flush=True
    )
    return kept


def filtered_train_pairs(index, split, vq_min=0.5, covis_floor=8, verbose=True):
    """The full pipeline: (covis-floored train pairs, unfiltered eval pairs)."""
    train_pairs, eval_pairs = build_train_pairs(index, split, vq_min, verbose)
    return apply_covis_floor(train_pairs, split, covis_floor), eval_pairs


class R2PairDataset(torch.utils.data.Dataset):
    """desc (cached) + images + GT warps/covis at /16 and /4, both dirs.

    GT grids subsample the `size`-px warp at patch-center pixels (offset 8
    stride 16, offset 2 stride 4; centers are 0.5 px off the continuous
    patch center — negligible at feature granularity).
    """

    def __init__(self, pairs, index, size=320, desc_cache=None):
        self.pairs = pairs
        self.index = index
        self.size = size
        self.desc_cache = Path(desc_cache) if desc_cache else DESC_CACHE_DIR

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        out = {"key": p["key"]}
        for tag, path in [("A", p["anchor"]), ("B", p["other"])]:
            out[f"desc_{tag}"] = torch.load(
                self.desc_cache / f"{frame_key(path)}.pt",
                map_location="cpu",
                weights_only=True,
            )["desc"].float()
            out[f"img_{tag}"] = img_to_tensor01(Image.open(path), self.size)[0]
        wab, cab, wba, cba = compute_warp_pair(
            self.index, p["anchor"], p["other"], self.size
        )
        for tag, w, c in [("ab", wab, cab), ("ba", wba, cba)]:
            out[f"warp16_{tag}"] = w[8::16, 8::16]
            out[f"covis16_{tag}"] = c[8::16, 8::16]
            out[f"warp4_{tag}"] = w[2::4, 2::4]
            out[f"covis4_{tag}"] = c[2::4, 2::4]
        return out


def collate(items):
    out = {"key": [b["key"] for b in items]}
    for k in items[0]:
        if k != "key":
            out[k] = torch.stack([b[k] for b in items])
    return out


def to_device(batch, device):
    """Batch (or single unsqueezed sample) -> device, non-tensors untouched."""
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def as_batch(sample, device):
    """Single dataset item -> batch of 1 on `device`."""
    return {
        k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
        for k, v in sample.items()
    }
