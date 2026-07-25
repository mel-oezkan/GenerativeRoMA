"""Held-out recon eval of the train-time decoders (docs/R2.md).

    python experiments/r2_recon_eval.py run=r2_v1 arm=joint step=12000

Fair comparison of the equal-budget arms: v1 joint (deep) vs desc-joint vs
desc-recon (RAE). See src/r2/recon_eval.py for what the A/B columns mean.

Output: results/<run>/<arm>/recon_eval.json
Configs: configs/recon_eval.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import echo, hydra_main, require  # noqa: E402
from src.r2.recon_eval import recon_eval  # noqa: E402


@hydra_main("recon_eval")
def main(cfg):
    require(cfg, "run", "arm")
    echo(cfg, f"recon eval {cfg.run}/{cfg.arm}")
    recon_eval(cfg)


if __name__ == "__main__":
    main()
