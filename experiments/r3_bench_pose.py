"""Out-of-domain matching eval: two-view pose AUC on MegaDepth-1500 /
ScanNet-1500 for any R2/R3 checkpoint (the "does it transfer" axis).

    python experiments/r3_bench_pose.py run=r2_v1 arm=joint step=12000
    python experiments/r3_bench_pose.py run=r3_ft arm=pretrained \
        model.refiners=true

Our arms are trained only on object-centric CO3D turntables at 320 px with
no refiners, so absolute AUC here is expected to be poor; the question is
whether the *relative* ordering (joint vs match, deep vs frozen-DINOv3)
survives a regime shift to scene-level wide-baseline pairs.

Model construction (turbo @ 320, amp off, refiner-free match()) lives in
src/r2/released.py; the benchmark loop is upstream's.

Output: results/<run>/<arm>/bench_<benchmark>[_refined].json
Configs: configs/bench_pose.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import echo, hydra_main, require  # noqa: E402
from src.r2.bench import bench_pose  # noqa: E402


@hydra_main("bench_pose")
def main(cfg):
    require(cfg, "arm")
    echo(cfg, f"pose benchmark {cfg.run}/{cfg.arm} on {cfg.benchmark}")
    bench_pose(cfg)


if __name__ == "__main__":
    main()
