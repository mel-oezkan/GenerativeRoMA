"""R2/R3: per-frame DINOv3 descriptor cache for the train/eval pairs.

    python experiments/r2_precompute_desc.py split=r3-4cat shard=0/2

Uses the full pretrained RoMaV2's descriptor (identical to R1's desc) — the
descriptor is frozen in training, so pretrained-ness is by design. Run the
covisibility census first: the full filter pipeline is applied before
caching, so only frames that can actually appear in training are extracted.

Cache layout and the reason it is per-frame: src/r2/precompute.py.
Configs: configs/precompute_desc.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import echo, hydra_main  # noqa: E402
from src.r2.precompute import precompute_desc  # noqa: E402


@hydra_main("precompute_desc")
def main(cfg):
    echo(cfg, f"desc precompute {cfg.split} shard={cfg.shard}")
    precompute_desc(cfg)


if __name__ == "__main__":
    main()
