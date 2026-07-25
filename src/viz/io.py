"""Reading results back out for the figure scripts."""

import json
from pathlib import Path

from src.paths import FIGURES_DIR, RESULTS_DIR  # noqa: F401  (re-exported)


def load_json(path):
    """Parsed JSON, or None if the file isn't there (figures skip inputs
    that haven't been produced yet)."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def load_result(rel):
    """load_json relative to results/ — e.g. load_result('r3_ft/match/metrics.json')."""
    return load_json(RESULTS_DIR / rel)


def run_config(run_dir):
    """The `config` block of results/<...>/run.json ({} if absent).

    Lets a figure read the run's actual hyperparameters (steps, GAN phase
    starts, ...) instead of hardcoding the values the run was launched with.
    """
    payload = load_json(Path(run_dir) / "run.json")
    return (payload or {}).get("config", {})


def fig_dir(name=None):
    """docs/figures[/<name>], created on demand."""
    out = FIGURES_DIR / name if name else FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out
