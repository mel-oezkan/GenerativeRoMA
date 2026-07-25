"""Training-loss curves for every run, parsed from the train logs.

    python experiments/visualizations/plot_losses.py

One figure per run dir with a panel per loss term and one line per arm
(comparable within a run), plus a per-arm figure in each arm's results
subdir. Two log formats are covered — the R1 RAE probe (recon terms only)
and the R2/R3 matcher training (matching terms prepended); which one a run
uses is `kind` in configs/figures/loss_curves.yaml.

The arm name comes from the log *filename* (train_<arm>.log), not the [tag]
— mv_selfpair logs tag themselves [mv]. Duplicate steps from resume replays
keep the last occurrence, and terms that are identically zero (an inactive
loss for that arm) are dropped rather than drawn as a flat line. Step counts
and GAN phase boundaries come from each arm's run.json, so the annotations
describe the run rather than the defaults it was launched with.

Re-run any time; figures are rebuilt from whatever the logs contain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import hydra_main  # noqa: E402
from src.paths import RESULTS_DIR  # noqa: E402
from src.viz import logs as vlogs  # noqa: E402
from src.viz.io import run_config  # noqa: E402
from src.viz.style import (  # noqa: E402
    ARM_COLORS,
    AXIS,
    FEATURE_COLORS,
    INK,
    MUTED,
    SURFACE,
    line_axes,
    pyplot,
)

plt = pyplot(report_style=False)

FORMATS = {
    "probe": (vlogs.PROBE_RE, vlogs.PROBE_TERMS, True),
    "train": (vlogs.TRAIN_RE, vlogs.TRAIN_TERMS, False),
}
# arm -> color across every figure; R1 feature arms and R2 training arms
# never share a figure, so one lookup serves both
COLORS = {**FEATURE_COLORS, **ARM_COLORS}
PHASE_TERMS = ("gan", "lam", "d")


def phases_for(arm_dir, fallback):
    """(steps, disc_start, gan_start) from the arm's run.json when present.

    Handles both schemas: the flat pre-config one (steps/disc_upd_start/
    gan_start) and the nested config dump (optim.steps, disc_start...).
    """
    cfg = run_config(arm_dir)
    steps = cfg.get("optim", {}).get("steps") or cfg.get("steps")
    disc = cfg.get("disc_upd_start", cfg.get("disc_start"))
    gan = cfg.get("gan_start")
    return (
        steps or fallback["steps"],
        fallback["disc_start"] if disc is None else disc,
        fallback["gan_start"] if gan is None else gan,
    )


def plot_series(ax, series, term, gan_start, color, label=None, hide_prefix=True):
    """One arm's line for one term, hiding the pre-GAN flat prefix."""
    steps, vals = series[term]
    if hide_prefix and term in ("gan", "lam"):
        pts = [(s, v) for s, v in zip(steps, vals) if s >= gan_start]
        if not pts:
            return
        steps, vals = zip(*pts)
    ax.plot(steps, vals, color=color, lw=2, label=label, zorder=2)


def style_panel(ax, term, terms, gan_phases, xmax, disc_start, gan_start):
    if gan_phases and term in PHASE_TERMS:
        x = disc_start if term == "d" else gan_start
        ax.axvline(x, color=AXIS, lw=1, ls="--", zorder=1)
        ax.text(x, 1.0, f" from {x}", transform=ax.get_xaxis_transform(),
                fontsize=7, color=MUTED, va="top")
    line_axes(ax, terms[term], xlim=(0, xmax))


def _panels(n):
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.2), facecolor=SURFACE)
    return fig, ([axes] if n == 1 else list(axes))


def plot_arm(run_dir, arm, series, terms, title, gan_phases, phases):
    """One figure for a single arm, saved into its results subdir."""
    out = run_dir / arm / "loss_curves.png"
    if not out.parent.is_dir():
        return None
    arm_terms = [t for t in terms if t in series]
    xmax, disc_start, gan_start = phases
    fig, axes = _panels(len(arm_terms))
    for ax, term in zip(axes, arm_terms):
        plot_series(ax, series, term, gan_start, COLORS.get(arm, MUTED),
                    hide_prefix=gan_phases)
        style_panel(ax, term, terms, gan_phases, xmax, disc_start, gan_start)
    # single series -> no legend; the title names the arm
    fig.suptitle(f"{title} — {arm} arm", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_dir(run_dir, arms, terms, title, gan_phases, phases):
    """The combined figure across arms."""
    if not arms:
        return None
    dir_terms = [t for t in terms if any(t in s for s in arms.values())]
    xmax, disc_start, gan_start = phases
    fig, axes = _panels(len(dir_terms))
    for ax, term in zip(axes, dir_terms):
        for arm in COLORS:  # fixed legend order regardless of glob order
            if arm in arms and term in arms[arm]:
                plot_series(ax, arms[arm], term, gan_start, COLORS[arm],
                            label=arm, hide_prefix=gan_phases)
        for arm in arms:  # arms without a palette slot
            if arm not in COLORS and term in arms[arm]:
                plot_series(ax, arms[arm], term, gan_start, MUTED, label=arm,
                            hide_prefix=gan_phases)
        style_panel(ax, term, terms, gan_phases, xmax, disc_start, gan_start)

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:  # some arms may lack the first panel's term
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=max(len(labels), 1),
               frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = run_dir / "loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


@hydra_main("figures/loss_curves")
def main(cfg):
    fallback = dict(cfg.fallback)
    for spec in cfg.runs:
        run_dir = RESULTS_DIR / spec.dir
        if not run_dir.is_dir():
            continue
        line_re, terms, default_phases = FORMATS[spec.kind]
        gan_phases = spec.get("gan_phases", default_phases)
        arms = vlogs.parse_arm_logs(run_dir, line_re, terms)
        # the run-level figure annotates with the first arm that records
        # phases; per-arm figures use their own
        phases = phases_for(run_dir / next(iter(arms), ""), fallback)
        out = plot_dir(run_dir, arms, terms, spec.title, gan_phases, phases)
        print(f"{spec.dir}: {out if out else 'no parseable train logs'}")
        if cfg.per_arm:
            for arm, series in arms.items():
                out = plot_arm(run_dir, arm, series, terms, spec.title,
                               gan_phases, phases_for(run_dir / arm, fallback))
                print(f"{spec.dir}/{arm}: {out if out else 'no results subdir'}")


if __name__ == "__main__":
    main()
