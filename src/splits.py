"""Split registry: sequence split + deterministic pair sampling.

Single source of truth for "which sequences and which pairs", shared by the
R1 probe, the R2/R3 matcher training, the covisibility census and the
precompute passes. Definitions live in configs/splits.yaml.

Everything here is seed-deterministic: pair keys are ``cat_seq_i_j`` and the
per-sequence sampling depends only on (seed, category, sequence), so caches
computed for one split are reused by any other split sharing the pair.
"""

import random

from omegaconf import OmegaConf

from src.data import sample_view_pairs, split_seqs
from src.paths import CONFIG_DIR, META_CACHE_DIR

SEED = 0

_cfg = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "splits.yaml"))
_DEFAULTS = _cfg["defaults"]
SPLITS = {name: {**_DEFAULTS, **spec} for name, spec in _cfg["splits"].items()}


def split_spec(split):
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}; have {sorted(SPLITS)}")
    return SPLITS[split]


def categories_for(split):
    """Category list of a split (never None: resolves the default set)."""
    cats = split_spec(split)["categories"]
    if cats is None:
        from src.data import EXTRACTED_CATEGORIES

        return list(EXTRACTED_CATEGORIES)
    return list(cats)


def covis_cache_for(split):
    """Path of the r2_pair_covis census for `split`."""
    name = split_spec(split).get("covis_cache") or f"pair_covis_{split}.json"
    return META_CACHE_DIR / name


def build_pairs(seqs_by_cat, n_per_seq, seed=SEED, self_pair=False):
    """Deterministic pairs, alternating near-/far-view conditioning image.

    self_pair keeps the same sampled anchors but conditions on the anchor
    itself (A == B); anchors drawn twice collapse via the key dedup."""
    pairs, seen = [], set()
    for cat, seqs in seqs_by_cat.items():
        for seq_name, frames in seqs:
            rng = random.Random(f"{seed}_{cat}_{seq_name}")
            for k, s in enumerate(sample_view_pairs(frames, n_per_seq, rng=rng)):
                i, j_near, j_far = s["idx"]
                j, other = (j_near, s["near"]) if k % 2 == 0 else (j_far, s["far"])
                if self_pair:
                    j, other = i, s["anchor"]
                key = f"{cat}_{seq_name}_{i}_{j}"
                if key not in seen:
                    seen.add(key)
                    pairs.append({"key": key, "anchor": s["anchor"], "other": other})
    return pairs


def get_splits(split, self_pair=False, seed=SEED):
    """(train_pairs, eval_pairs) for a named split."""
    import numpy as np

    cfg = split_spec(split)
    eval_seqs, train_seqs = split_seqs(
        categories=cfg["categories"],
        n_eval=cfg["n_eval_seqs"],
        n_train=cfg["n_train_seqs"],
        seed=seed,
    )
    if cfg["min_seq_vq"] is not None:  # eval-only splits: filter eval too
        from src.co3d_geom import load_seq_quality

        for cat, seqs in eval_seqs.items():
            vq = load_seq_quality(cat)
            eval_seqs[cat] = [
                s for s in seqs
                if not np.isnan(vq.get(s[0], float("nan")))
                and vq[s[0]] >= cfg["min_seq_vq"]
            ]
    train_pairs = build_pairs(train_seqs, cfg["train_pairs_per_seq"], seed, self_pair)
    eval_pairs = build_pairs(eval_seqs, cfg["eval_pairs_per_seq"], seed, self_pair)
    return train_pairs, eval_pairs
