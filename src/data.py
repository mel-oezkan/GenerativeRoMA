"""Minimal CO3D multiview access: sample frames and (near, far) view pairs.

CO3D sequences are video walks around an object, so frame index distance is a
proxy for viewpoint distance: adjacent frames ~ similar views, frames half a
sequence apart ~ opposite views.
"""

import random
from pathlib import Path

CO3D_ROOT = Path("/visinf/projects_students/dlcv2025_groupZ/co3d_full")
# Categories that are already extracted (not zipped).
EXTRACTED_CATEGORIES = ["hydrant", "car", "toybus"]
# Additional categories extracted for H1 (~24 sequences each, see docs/H1.md);
# kept separate so the MVP 0-5 scripts keep their original data/splits.
H1_NEW_CATEGORIES = ["bench", "toilet", "suitcase", "toytrain"]
H1_CATEGORIES = EXTRACTED_CATEGORIES + H1_NEW_CATEGORIES


def list_sequences(category, min_frames=60):
    cat_dir = CO3D_ROOT / category
    seqs = []
    for seq_dir in sorted(cat_dir.iterdir()):
        img_dir = seq_dir / "images"
        if not img_dir.is_dir():
            continue
        frames = sorted(img_dir.glob("frame*.jpg"))
        if len(frames) >= min_frames:
            seqs.append((seq_dir.name, frames))
    return seqs


def split_seqs(categories=None, n_eval=6, n_train=12, seed=0):
    """Deterministic eval/train sequence split (MVP 3 recipe). One rng is
    shared across categories, so the category order matters and adding a
    category WOULD reshuffle existing splits — that's why H1's new
    categories get their own per-category split (h1_scale_finetune.py)."""
    rng = random.Random(seed)
    eval_seqs, train_seqs = {}, {}
    for cat in categories or EXTRACTED_CATEGORIES:
        seqs = list_sequences(cat)
        rng.shuffle(seqs)
        eval_seqs[cat] = seqs[:n_eval]
        train_seqs[cat] = seqs[n_eval:n_eval + n_train]
    return eval_seqs, train_seqs


def sample_view_pairs(frames, n_pairs, near_offset=2, rng=None):
    """Return list of dicts with anchor / near / far frame paths.

    near = `near_offset` frames away (similar viewpoint),
    far  = half the sequence away (very different viewpoint).
    """
    rng = rng or random.Random(0)
    n = len(frames)
    pairs = []
    for _ in range(n_pairs):
        i = rng.randrange(0, n - near_offset)
        j_near = i + near_offset
        j_far = (i + n // 2) % n
        pairs.append(
            {"anchor": frames[i], "near": frames[j_near], "far": frames[j_far],
             "idx": (i, j_near, j_far)}
        )
    return pairs
