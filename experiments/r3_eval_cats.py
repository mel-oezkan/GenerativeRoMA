"""Per-category matching eval of any trained R2/R3 checkpoint.

    python experiments/r3_eval_cats.py run=r2_v1 arm=joint step=12000
    python experiments/r3_eval_cats.py arm=pretrained-refined
    python experiments/r3_eval_cats.py run=r2_v1 arm=joint step=12000 \
        split=co3d-unseen tag=co3d-unseen

Same rows/metrics as the training-time eval (src/r2/metrics.py), kept per
CO3D category so the aggregate breaks down by category and near/far.

Reference arms:
  arm=pretrained          the stock romav2.pt matcher, zero-shot through the
                          same coarse-only /4-warp pipeline. Our arms have no
                          refiners, so this is the like-for-like row and the
                          natural "what does fine-tuning buy" baseline.
  arm=pretrained-refined  ...plus the released ConvRefiner stack (patch
                          4/2/1, turbo @ 320) = the full published system;
                          the headroom the refiners buy. Reports the
                          pre-refiner warp too (epe_matcher), so the +/-
                          refiner comparison comes from one run.

Output: results/<run>/<arm>/metrics_per_cat[_<tag>].json
Configs: configs/eval_cats.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import echo, hydra_main, require  # noqa: E402
from src.r2.metrics import per_category_eval  # noqa: E402


@hydra_main("eval_cats")
def main(cfg):
    require(cfg, "arm")
    echo(cfg, f"per-category eval {cfg.run}/{cfg.arm} [{cfg.split}]")
    per_category_eval(cfg)


if __name__ == "__main__":
    main()
