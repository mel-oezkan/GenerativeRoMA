"""Pair covisibility census at /16 (CPU, parallel).

For every frame/pose-quality-filtered train pair, counts covisible tokens on
the g x g grid in both directions and writes

    META_CACHE_DIR/pair_covis_<split>.json   {key: [n_ab, n_ba]}

Training drops pairs whose *better* direction is below the covis floor —
those carry no matching signal, and their confidence-BCE would supervise
"not covisible" where the truth is merely "unknown" (missing depth). Eval
pairs are censused (for reporting) but never filtered.

Pair keys are seed-deterministic, so counts computed for one split are
reused by any other split that shares the pair (the r3-4cat census seeds
itself from the hydrant one).
"""

import json
from multiprocessing import Pool

import numpy as np

from src.co3d_geom import compute_warp_pair, load_frame_index_multi
from src.r2.dataset import build_train_pairs
from src.splits import SPLITS, categories_for, covis_cache_for

# worker-global: the frame index is built once per process, not per pair
_index = None
_cats = None
_size = 320


def _init(cats, size):
    global _index, _cats, _size
    _cats, _size = cats, size
    _index = load_frame_index_multi(cats)


def _count(pair):
    _, cab, _, cba = compute_warp_pair(_index, pair["anchor"], pair["other"], _size)
    return pair["key"], [
        int(cab[8::16, 8::16].sum()),
        int(cba[8::16, 8::16].sum()),
    ]


def seed_from_other_splits(split, wanted, done):
    """Reuse counts from other splits' censuses (identical keys => same pair)."""
    for other in SPLITS:
        cache = covis_cache_for(other)
        if other != split and cache.exists():
            prev = json.loads(cache.read_text())
            done.update({k: v for k, v in prev.items()
                         if k in wanted and k not in done})
    return done


def census(cfg):
    cats = categories_for(cfg.split)
    out = covis_cache_for(cfg.split)
    index = load_frame_index_multi(cats)
    train_pairs, eval_pairs = build_train_pairs(index, cfg.split, cfg.data.vq_min)
    pairs = train_pairs + eval_pairs
    wanted = {p["key"] for p in pairs}

    done = json.loads(out.read_text()) if out.exists() else {}
    done = seed_from_other_splits(cfg.split, wanted, done)
    todo = [p for p in pairs if p["key"] not in done]
    print(f"{len(todo)} pairs to census ({len(done)} cached)", flush=True)

    with Pool(cfg.workers, initializer=_init, initargs=(cats, cfg.size)) as pool:
        for n, (key, counts) in enumerate(
            pool.imap_unordered(_count, todo, cfg.chunk)
        ):
            done[key] = counts
            if n % cfg.save_every == 0:
                print(f"  {n}/{len(todo)}", flush=True)
                out.write_text(json.dumps(done))
    out.write_text(json.dumps(done))

    counts = [max(v) for v in done.values()]
    print(f"census done: n={len(counts)} median={np.median(counts):.0f} "
          f"p10={np.percentile(counts, 10):.0f} "
          f"frac<8={float(np.mean(np.array(counts) < 8)):.3f}", flush=True)
    return out
