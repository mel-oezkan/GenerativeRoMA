"""Two-view pose benchmarks (MegaDepth-1500 / ScanNet-1500).

The released RoMaV2 class implements match/sample/pose extraction, so the
benchmark itself is upstream's; this module only builds the model with our
weights and records the result.

Output: results/<run>/<arm>/bench_<benchmark>[_refined].json
"""

import json

from src.paths import BENCH_ROOT, RESULTS_DIR
from src.r2.released import benchmark_model

AUC_KEYS = ("auc_5", "auc_10", "auc_20")


def build_benchmark(name, limit=None):
    if name == "mega1500":
        from romav2.benchmarks.mega1500 import Mega1500

        bench = Mega1500(data_root=str(BENCH_ROOT / "megadepth"))
        if limit:  # smoke test; NpzFile is read-only, swap in plain dicts
            bench.scenes = [
                {k: (s[k][:limit] if k == "pair_infos" else s[k])
                 for k in ("pair_infos", "intrinsics", "poses", "image_paths")}
                for s in bench.scenes
            ]
        return bench
    if name == "scannet1500":
        from romav2.benchmarks.scannet1500 import ScanNet1500

        return ScanNet1500(data_root=str(BENCH_ROOT / "scannet/scans"))
    raise ValueError(f"unknown benchmark {name!r}")


def bench_pose(cfg):
    model = benchmark_model(cfg.run, cfg.arm, cfg.step, cfg.model.desc_only,
                            cfg.model.refiners)
    bench = build_benchmark(cfg.benchmark, cfg.limit)

    name = f"{cfg.run}/{cfg.arm}"
    out = {k: float(v) for k, v in bench.benchmark(model, model_name=name).items()}
    out["config"] = {
        "run": cfg.run, "arm": cfg.arm, "step": cfg.step,
        "benchmark": cfg.benchmark, "setting": "turbo@320",
        "refiners": cfg.model.refiners, "desc_only": cfg.model.desc_only,
    }
    out_dir = RESULTS_DIR / cfg.run / cfg.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.benchmark + ("_refined" if cfg.model.refiners else "")
    (out_dir / f"bench_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"{name} [{tag}]: "
          + " ".join(f"{k} {out[k]:.4f}" for k in AUC_KEYS))
    return out
