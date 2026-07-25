"""Feature caches the R2/R3 pipeline reads from.

Two passes, both resumable (existing files are skipped) and shardable with
"i/n" so one process per GPU can run in parallel:

  desc  per-frame DINOv3 descriptors for the train/eval pairs
        feats.desc/<cat>_<seq>_<frame_stem>.pt
            {"desc": (2048, g, g) fp16, "path": <image path>}
        2048 = the two Descriptor layers (1024 each) channel-concatenated in
        list order; split back into [.. :1024], [1024: ..] for the Matcher's
        f_list input. R1 cached desc per *pair* (view A only); the matcher
        needs both views and frames repeat across pairs, hence per frame.

  mv    mv_vit output tokens of view A from a *trained* checkpoint, in the
        R1 pair-cache format, so the R1 probe can measure it
        feats.root/<cache_name>/<arm>/<key>.pt
            {"mv": (1024, g, g) fp16, "anchor": <image path>}
"""

from pathlib import Path

import torch
from PIL import Image

from src.co3d_geom import load_frame_index_multi
from src.paths import DESC_CACHE_DIR, FEATS_ROOT
from src.r2.dataset import filtered_train_pairs, frame_key
from src.r2.model import load_trained_model
from src.romav2_utils import img_to_tensor01, load_romav2_frozen
from src.splits import categories_for, get_splits


def unique_frames(pairs):
    """{frame_key: image path} over both views of every pair."""
    frames = {}
    for p in pairs:
        for f in (p["anchor"], p["other"]):
            frames[frame_key(f)] = f
    return frames


def shard_of(items, shard):
    if not shard:
        return items
    i, n = map(int, shard.split("/"))
    return items[i::n]


def _extract_desc(model, path, size, device):
    img = img_to_tensor01(Image.open(path), size).to(device)
    f_list = model.f(img)  # tuple of (1, g, g, 1024)
    return torch.cat(f_list, dim=-1)[0].permute(2, 0, 1).contiguous()


def write_desc_cache(todo, cache_dir, size=320, device="cuda", log_every=200):
    """Extract and save descriptors for [(key, path)] with the frozen model."""
    if not todo:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = load_romav2_frozen()
    with torch.no_grad():
        for n, (key, path) in enumerate(todo):
            desc = _extract_desc(model, path, size, device)
            torch.save({"desc": desc.half().cpu(), "path": str(path)},
                       cache_dir / f"{key}.pt")
            if n % log_every == 0:
                print(f"  {n}/{len(todo)}", flush=True)


def precompute_desc(cfg):
    """Per-frame desc cache over the *filtered* pair set (same frames the
    training run will read; eval pairs stay unfiltered, matching train)."""
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else DESC_CACHE_DIR
    index = load_frame_index_multi(categories_for(cfg.split))
    train_pairs, eval_pairs = filtered_train_pairs(
        index, cfg.split, cfg.data.vq_min, cfg.data.covis_floor
    )
    frames = shard_of(sorted(unique_frames(train_pairs + eval_pairs).items()),
                      cfg.shard)
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [(k, f) for k, f in frames if not (cache_dir / f"{k}.pt").exists()]
    print(f"{len(frames)} frames in shard, {len(todo)} to compute", flush=True)
    write_desc_cache(todo, cache_dir, cfg.size, cfg.device, cfg.log_every)
    print("desc cache complete", flush=True)


def fill_desc(cfg):
    """Descs for the *probe* pairs that the training filter left uncached.

    The probe deliberately runs on unfiltered pairs, so it reaches frames the
    training cache never needed. Run this once before the arms — two
    extraction processes would otherwise race on the same file.
    """
    cache_dir = Path(cfg.cache_dir) if cfg.get("cache_dir") else DESC_CACHE_DIR
    train_pairs, eval_pairs = get_splits(cfg.split)
    frames = unique_frames(train_pairs + eval_pairs)
    todo = sorted((k, f) for k, f in frames.items()
                  if not (cache_dir / f"{k}.pt").exists())
    print(f"{len(frames)} frames, {len(todo)} missing desc", flush=True)
    write_desc_cache(todo, cache_dir, cfg.size, cfg.device, cfg.log_every)
    print("desc fill complete", flush=True)


def extract_mv(cfg):
    """Cache mv tokens of a trained arm for the R1 probe pairs."""
    desc_cache = Path(cfg.cache_dir) if cfg.get("cache_dir") else DESC_CACHE_DIR
    train_pairs, eval_pairs = get_splits(cfg.split)
    pairs = shard_of(train_pairs + eval_pairs, cfg.shard)
    out_dir = FEATS_ROOT / cfg.cache_name / cfg.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pairs if not (out_dir / f"{p['key']}.pt").exists()]
    print(f"[{cfg.arm}] {len(pairs)} pairs in shard, {len(todo)} to extract",
          flush=True)
    if not todo:
        return
    model = load_trained_model(cfg.run, cfg.arm, cfg.step, device=cfg.device)
    with torch.no_grad():
        for n, p in enumerate(todo):
            desc = [
                torch.load(desc_cache / f"{frame_key(f)}.pt", map_location="cpu",
                           weights_only=True)["desc"].float()[None].to(cfg.device)
                for f in (p["anchor"], p["other"])
            ]
            img_A = img_to_tensor01(Image.open(p["anchor"]), cfg.size).to(cfg.device)
            img_B = img_to_tensor01(Image.open(p["other"]), cfg.size).to(cfg.device)
            preds = model(desc[0], desc[1], img_A, img_B)
            mv = preds["mv_A"][0].permute(2, 0, 1).contiguous()  # (1024,g,g)
            torch.save({"mv": mv.half().cpu(), "anchor": str(p["anchor"])},
                       out_dir / f"{p['key']}.pt")
            if n % cfg.log_every == 0:
                print(f"  [{cfg.arm}] {n}/{len(todo)}", flush=True)
    print(f"[{cfg.arm}] extraction complete", flush=True)
