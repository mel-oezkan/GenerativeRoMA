"""R2/R3 matcher training: joint matching + reconstruction (docs/R2.md).

    python experiments/r2_train.py experiment=r2_v1 arm=joint

Stock Matcher architecture (mv_vit vit_base, DPT head -> warp+conf @ /4),
frozen precomputed DINOv3 desc inputs (r2_precompute_desc.py), 320 px, no
refiners. Losses per direction (bidirectional):
  - attn CE @ /16: cross-entropy of the global correlation logits against
    the GT match cell, soft-binned bilinearly over 4 neighbor cells
    (RoMa-style regression-by-classification), covisible tokens only
  - warp Charbonnier @ /4 + BCE(confidence, covisibility)
  - recon: RAEDecoder on the mv tokens of BOTH views -> images A and B,
    L1 + LPIPS(VGG) averaged over the two views; the *measurement* remains
    the post-hoc R1 probe protocol

GT warps come from CO3D depth+pose on the fly (src/co3d_geom.py, validated
by r2_warp_check.py). Eval: EPE/PCK of the /4 warp + coarse argmax EPE @
/16 on held-out pairs.

Configs: configs/r2_train.yaml (recipe) + configs/experiment/<run>.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import echo, hydra_main, require  # noqa: E402
from src.r2.train import train  # noqa: E402


@hydra_main("r2_train")
def main(cfg):
    require(cfg, "arm", "run")
    echo(cfg, f"r2_train {cfg.run}/{cfg.arm}")
    train(cfg)


if __name__ == "__main__":
    main()
