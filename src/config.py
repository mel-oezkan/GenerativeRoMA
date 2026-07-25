"""Hydra glue shared by the entry points in experiments/.

Every script is the same three lines — compose the config, echo it into the
log, hand it to a function in src/ — so that lives here instead of being
copy-pasted eight times.
"""

import hydra
from omegaconf import OmegaConf

from src.paths import CONFIG_DIR


def hydra_main(config_name):
    """@hydra_main("r2_train") on a `main(cfg)` function.

    config_path is absolute so a script runs identically from any working
    directory (the drivers cd to the repo root, interactive use often does
    not, and chdir is disabled — see configs/hydra_defaults.yaml).
    """
    return hydra.main(
        version_base="1.3", config_path=str(CONFIG_DIR), config_name=config_name
    )


def echo(cfg, title=None):
    """Print the fully composed config — the log then records exactly what
    ran, including which experiment file and overrides produced it."""
    if title:
        print(f"--- {title} ---", flush=True)
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    return cfg


def require(cfg, *keys):
    """Fail fast on missing mandatory values, with a usable message."""
    missing = [k for k in keys if OmegaConf.select(cfg, k) in (None, "???")]
    if missing:
        raise ValueError(
            "missing required config value(s): "
            + ", ".join(f"{k}=<...>" for k in missing)
        )
    return cfg
