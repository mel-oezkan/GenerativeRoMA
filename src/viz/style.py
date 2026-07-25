"""One palette and one rcParams block for every figure in the repo.

Reference palette (dataviz skill), light mode. Colors are assigned by
*meaning* — an arm keeps its color across every figure — so import the
named constants rather than re-typing hex codes.
"""

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
AXIS = BASE  # historical alias used by the loss-curve scripts
GRID_SOFT = "#e6e5e0"  # lighter grid, used on the depth/census histograms

# categorical slots, validated order
SLOTS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#8a63d2"]
ALERT = "#d1495b"

# --- semantic assignments -------------------------------------------------
C_MATCH = SLOTS[0]   # matching-only arm
C_JOINT = SLOTS[1]   # joint (matching + recon) arm
C_RECON = SLOTS[5]   # recon-only arm
C_PRE = MUTED        # reference model, not an arm

# R2/R3 training arms
ARM_COLORS = {"match": C_MATCH, "joint": C_JOINT, "recon": C_RECON,
              "pretrained": C_PRE}
# R1 probe arms (feature taps)
FEATURE_COLORS = {
    "desc": SLOTS[0],
    "mv": SLOTS[1],
    "dpt": SLOTS[2],
    "mvdesc": SLOTS[3],
    "mv_selfpair": SLOTS[4],
}

RC_PARAMS = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "legend.frameon": False,
}


def pyplot(report_style=True):
    """Agg-backed pyplot, optionally with the report rcParams applied.

    Figure scripts run headless on the cluster, so the backend is forced
    before pyplot is imported.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if report_style:
        plt.rcParams.update(RC_PARAMS)
    return plt


def agg_figure(nrows=1, ncols=1, report_style=False, **kwargs):
    """(plt, fig, axes) — image grids default to matplotlib's plain look."""
    plt = pyplot(report_style)
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    return plt, fig, axes


def bar_labels(ax, rects, fmt="{:.2f}"):
    """Value labels above a bar container."""
    for r in rects:
        ax.annotate(
            fmt.format(r.get_height()),
            (r.get_x() + r.get_width() / 2, r.get_height()),
            ha="center", va="bottom", fontsize=8, color=INK2,
        )


def grouped(ax, groups, series, width=0.24, gap=0.02):
    """Grouped bars. series = [(label, color, values)]; groups = tick labels."""
    import numpy as np

    x = np.arange(len(groups))
    n = len(series)
    rects = []
    for i, (label, color, vals) in enumerate(series):
        off = (i - (n - 1) / 2) * (width + gap)
        rects.append(ax.bar(x + off, vals, width, label=label, color=color))
    ax.set_xticks(x, groups)
    ax.tick_params(length=0)
    return rects


def line_axes(ax, title=None, xlabel="step", xlim=None):
    """The loss-curve panel look (used by r1/r2_plot_losses)."""
    if title:
        ax.set_title(title, fontsize=10, color=INK)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    return ax
