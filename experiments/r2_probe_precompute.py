"""R2/R3 post-hoc probe, step 1: cache mv tokens of a trained matcher.

    python experiments/r2_probe_precompute.py fill_desc=true
    python experiments/r2_probe_precompute.py run=r3_ft arm=joint step=6000 \
        cache_name=r3_probe_320

Extracts the mv_vit output tokens of view A for every R1 probe pair
(*unfiltered*, identical to the stock-RoMaV2 probe, so PSNR is directly
comparable to it). Step 2 is r1_recon_probe.py feature=mv with cache_dir
and out_dir pointing here — same RAE recipe, fresh decoder per arm.

Configs: configs/probe_precompute.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import echo, hydra_main, require  # noqa: E402
from src.r2.precompute import extract_mv, fill_desc  # noqa: E402


@hydra_main("probe_precompute")
def main(cfg):
    echo(cfg, "probe precompute")
    if cfg.fill_desc:
        fill_desc(cfg)
        return
    require(cfg, "arm")
    extract_mv(cfg)


if __name__ == "__main__":
    main()
