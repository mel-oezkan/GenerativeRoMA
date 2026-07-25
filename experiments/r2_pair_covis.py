"""Pair covisibility census at /16 for the R2/R3 train pairs (CPU, parallel).

    python experiments/r2_pair_covis.py split=r3-4cat

Writes META_CACHE_DIR/pair_covis_<split>.json; see src/r2/census.py for what
the counts mean and how they are reused across splits.

Configs: configs/pair_covis.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import echo, hydra_main  # noqa: E402
from src.r2.census import census  # noqa: E402


@hydra_main("pair_covis")
def main(cfg):
    echo(cfg, f"covis census {cfg.split}")
    census(cfg)


if __name__ == "__main__":
    main()
