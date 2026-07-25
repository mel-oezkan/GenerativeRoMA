"""R1: can a small decoder reconstruct images from *frozen* features?

    python experiments/r1_recon_probe.py feature=mv split=hydrant-full
    python experiments/r1_recon_probe.py precompute_only=true shard=0/2

Baseline endpoint of the reconstructability/matching tradeoff (docs/R1.md):
the encoder stays frozen, only an RAEDecoder is trained on cached
pair-conditioned features. Recipe and its deviations: src/r1/probe.py.

Precompute the shared feature cache once with precompute_only=true, then
train one process per feature (cache-only, no RoMaV2 in memory).

Configs: configs/r1_probe.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import echo, hydra_main, require  # noqa: E402
from src.r1 import probe  # noqa: E402


@hydra_main("r1_probe")
def main(cfg):
    echo(cfg, "r1_recon_probe")
    if cfg.precompute_only:
        probe.precompute(cfg)
        return
    require(cfg, "feature")
    probe.train(cfg)


if __name__ == "__main__":
    main()
