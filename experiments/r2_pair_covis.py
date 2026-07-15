"""Pair covisibility census at /16 for the R2 train pairs (CPU, parallel).

For every (frame/seq-quality-filtered) train pair, counts covisible tokens
on the 20x20 grid in both directions and writes
    META_CACHE_DIR/pair_covis_hydrant.json   {key: [n_ab, n_ba]}
r2_train drops pairs whose *better* direction is below COVIS_FLOOR — those
carry no matching signal, and their confidence-BCE would supervise
"not covisible" where the truth is merely "unknown" (missing depth).
Eval pairs are included in the census (for reporting) but never filtered.
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.co3d_geom import META_CACHE_DIR, compute_warp_pair, load_frame_index

OUT = META_CACHE_DIR / "pair_covis_hydrant.json"
SIZE = 320

_index = None


def _init():
    global _index
    _index = load_frame_index("hydrant")


def _count(pair):
    _, cab, _, cba = compute_warp_pair(_index, pair["anchor"], pair["other"], SIZE)
    return pair["key"], [
        int(cab[8::16, 8::16].sum()),
        int(cba[8::16, 8::16].sum()),
    ]


def main():
    from experiments.r2_train import build_train_pairs

    index = load_frame_index("hydrant")
    train_pairs, eval_pairs = build_train_pairs(index)
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [p for p in train_pairs + eval_pairs if p["key"] not in done]
    print(f"{len(todo)} pairs to census ({len(done)} cached)", flush=True)
    with Pool(7, initializer=_init) as pool:
        for n, (key, counts) in enumerate(pool.imap_unordered(_count, todo, 16)):
            done[key] = counts
            if n % 500 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
                OUT.write_text(json.dumps(done))
    OUT.write_text(json.dumps(done))
    counts = [max(v) for k, v in done.items()]
    import numpy as np

    print(f"census done: n={len(counts)} median={np.median(counts):.0f} "
          f"p10={np.percentile(counts, 10):.0f} "
          f"frac<8={float(np.mean(np.array(counts) < 8)):.3f}", flush=True)


if __name__ == "__main__":
    main()
