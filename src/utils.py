"""Experiment bookkeeping helpers."""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_run_json(out_dir, config):
    """Drop run.json (command, git hash, date, config) into out_dir so every
    results/<name>/ directory is self-describing."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        ).stdout.strip()
    except OSError:
        rev = ""
    payload = {"argv": sys.argv, "git": rev,
               "date": time.strftime("%Y-%m-%d %H:%M:%S"), "config": config}
    Path(out_dir, "run.json").write_text(json.dumps(payload, indent=2))


def warmup_cosine_lambda(warmup, total, floor):
    """LambdaLR factor: linear ramp (k+1)/warmup, then cosine to `floor`
    (= eta_min/lr). Closed-form so resume only needs last_epoch. Same schedule
    as the H7b winner recipe (h7_gan_cotrain --warmup_steps)."""
    import math

    def fn(k):
        if k < warmup:
            return (k + 1) / warmup
        return floor + (1 - floor) * 0.5 * (
            1 + math.cos(math.pi * (k - warmup) / max(total - warmup, 1)))
    return fn
