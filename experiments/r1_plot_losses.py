"""Training-loss curves for the R1 recon-probe runs, parsed from train logs.

Each results dir (3cat v3, v1 conv, hydrant-full) gets one figure with a
panel per loss term (l1, lpips, G gan, lam, D) and one line per arm, so the
arms are directly comparable within a split. Log lines look like

    [mv] step 3600 l1 0.1853 lpips 0.6342 gan 0.9714 lam 0.212 d 0.2623

(v1 conv logs stop after lpips; gan/lam appear from step 0 in v3, d only
once the discriminator trains). The arm name comes from the log *filename*
(train_<arm>.log), not the [tag] — mv_selfpair logs tag themselves [mv].
Duplicate steps from resume replays keep the last occurrence. Re-run any
time; figures are rebuilt from whatever the logs currently contain.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# simplify the imports by appending the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.r1_recon_probe import DISC_UPD_START, GAN_START, RESULTS_DIR, STEPS


# regex to read out the values from the log + descriotions
LINE_RE = re.compile(
    r"\[\w+\] step (?P<step>\d+) l1 (?P<l1>[\d.]+) lpips (?P<lpips>[\d.]+)"
    r"(?: gan (?P<gan>[\d.]+) lam (?P<lam>[\d.]+))?(?: d (?P<d>[\d.]+))?"
)
TERMS = {
    "l1": "L1",
    "lpips": "LPIPS (VGG)",
    "gan": "G adversarial",
    "lam": "adaptive weight λ",
    "d": "D hinge",
}

# fixed arm -> color (reference palette slots 1-5, validated order); color
# follows the arm across every figure
ARM_COLORS = {
    "desc": "#2a78d6",
    "mv": "#1baf7a",
    "dpt": "#eda100",
    "mvdesc": "#008300",
    "mv_selfpair": "#4a3aa7",
}
RUN_DIRS = {  # results subdir -> (title, has GAN phases)
    "r1_recon_probe": ("R1 probe v3 (RAE recipe) — 3cat split", True),
    "r1_recon_probe_hydrant": ("R1 probe v3 (RAE recipe) — hydrant-full split", True),
    "r1_recon_probe_conv": ("R1 probe v1 (ConvDecoder, no GAN) — 3cat split", False),
}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def parse_log(path) -> dict[str, tuple[list[int], list[float]]]:
    """Helper function to parse the training logs.

    -> {term: (steps, values)} with resume duplicates collapsed (last wins)."""
    by_step = {}
    for line in path.read_text(errors="replace").splitlines():
        m = LINE_RE.search(line)
        if m:
            by_step[int(m["step"])] = {
                k: float(m[k]) for k in TERMS if m[k] is not None
            }
    series = {}
    for step in sorted(by_step):
        for k, v in by_step[step].items():
            series.setdefault(k, ([], []))
            series[k][0].append(step)
            series[k][1].append(v)
    return series


def plot_dir(run_dir, title, gan_phases):
    """Iterates automatically through the dir and plots the combined results"""
    arms = {}
    for log in sorted(run_dir.glob("train_*.log")):
        # cleaner labels
        arm = log.stem.removeprefix("train_")
        series = parse_log(log)

        if series:
            arms[arm] = series

    if not arms:
        return None

    terms = [t for t in TERMS if any(t in s for s in arms.values())]
    fig, axes = plt.subplots(
        1, len(terms), figsize=(3.4 * len(terms), 3.2), facecolor=SURFACE
    )

    axes = [axes] if len(terms) == 1 else list(axes)
    for ax, term in zip(axes, terms):
        ax.set_facecolor(SURFACE)
        if gan_phases and term in ("gan", "lam", "d"):
            ax.axvline(
                GAN_START if term != "d" else DISC_UPD_START,
                color=AXIS,
                lw=1,
                ls="--",
                zorder=1,
            )

        for arm, series in arms.items():
            if term not in series:
                continue
            steps, vals = series[term]
            # gan/lam are logged as 0 before GAN_START; hide the flat prefix
            if gan_phases and term in ("gan", "lam"):
                pts = [(s, v) for s, v in zip(steps, vals) if s >= GAN_START]
                if not pts:
                    continue
                steps, vals = zip(*pts)

            ax.plot(
                steps, vals, color=ARM_COLORS.get(arm, MUTED), lw=2, label=arm, zorder=2
            )

        ax.set_title(TERMS[term], fontsize=10, color=INK)
        ax.set_xlim(0, STEPS)
        ax.set_xlabel("step", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=8, colors=MUTED)
        
        ax.grid(True, color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)

    if gan_phases:
        for ax, term in zip(axes, terms):
            if term in ("gan", "lam", "d"):
                # Todo: this should not be fixed we should be able to read it from log or some conf
                x = GAN_START if term != "d" else DISC_UPD_START
                ax.text(
                    x,
                    1.0,
                    f" from {x}",
                    transform=ax.get_xaxis_transform(),
                    fontsize=7,
                    color=MUTED,
                    va="top",
                )

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:  # some arms may lack the first panel's term
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        bbox_to_anchor=(0.5, -0.04),
    )
    
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = run_dir / "loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    for name, (title, gan_phases) in RUN_DIRS.items():
        run_dir = RESULTS_DIR / name
        if not run_dir.is_dir():
            continue
        out = plot_dir(run_dir, title, gan_phases)
        print(f"{name}: {out if out else 'no parseable train logs'}")


if __name__ == "__main__":
    main()
