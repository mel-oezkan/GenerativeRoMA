"""Filesystem roots for the R-line, resolved once from configs/paths.yaml.

Import-time constants rather than a config object on purpose: the library
modules (co3d_geom, romav2_utils, data) and the figure scripts are used
outside Hydra entry points, and every one of them wants the same roots.
Set GENROMA_PATHS to an alternative YAML to relocate the data.
"""

import os
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
_PATHS_YAML = Path(os.environ.get("GENROMA_PATHS", CONFIG_DIR / "paths.yaml"))

_cfg = OmegaConf.load(_PATHS_YAML)
_cfg.repo = str(REPO_ROOT)
_cfg.home = str(Path.home())
PATHS = OmegaConf.to_container(_cfg, resolve=True)

# --- CO3D -----------------------------------------------------------------
CO3D_ROOTS = [Path(p) for p in PATHS["co3d"]["roots"]]
CO3D_ROOT = CO3D_ROOTS[0]

# --- feature / metadata caches --------------------------------------------
FEATS_ROOT = Path(PATHS["feats"]["root"])
META_CACHE_DIR = Path(PATHS["feats"]["meta"])
R1_CACHE_DIR = Path(PATHS["feats"]["r1"])
DESC_CACHE_DIR = Path(PATHS["feats"]["desc"])

# --- benchmarks -----------------------------------------------------------
BENCH_ROOT = Path(PATHS["bench_root"])

# --- pretrained weights ---------------------------------------------------
ROMAV2_CKPT = Path(PATHS["checkpoints"]["romav2"])
DINO_S8_CKPT = Path(PATHS["checkpoints"]["dino_s8"])

# --- inside the repo ------------------------------------------------------
ROMAV2_SRC = Path(PATHS["repo_paths"]["romav2_src"])
RESULTS_DIR = Path(PATHS["repo_paths"]["results"])
FIGURES_DIR = Path(PATHS["repo_paths"]["figures"])
DOCS_DIR = Path(PATHS["repo_paths"]["docs"])


def root_for(category):
    """Download root holding `category` (first match wins, see CO3D_ROOTS)."""
    for r in CO3D_ROOTS:
        if (r / category / "frame_annotations.jgz").exists():
            return r
    raise FileNotFoundError(f"category {category!r} in none of {CO3D_ROOTS}")
