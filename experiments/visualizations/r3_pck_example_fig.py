"""Didactic figure: EPE vs PCK@1 / PCK@5 on two toy 12-pixel models.

Model A is accurate on most pixels but has two blowouts; model B is
uniformly mediocre. A wins on PCK, B wins on EPE -- the point of
reporting both. Distances are in pixels at 320 px, the r3-4cat scale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from src.viz.io import fig_dir  # noqa: E402
from src.viz.style import ALERT, INK, INK2, MUTED, SLOTS, SURFACE, pyplot  # noqa: E402

plt = pyplot()
OUT = fig_dir("r3") / "fig_epe_pck_example.png"

C_A, C_B = SLOTS[0], SLOTS[1]
C_HIT1, C_HIT5, C_MISS = SLOTS[1], SLOTS[0], ALERT

D_A = np.array([0.4, 0.6, 0.8, 1.2, 1.5, 2.0, 2.4, 3.1, 4.2, 4.8, 30.0, 45.0])
D_B = np.array([3.0, 4.0, 4.6, 4.9, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
ANG = np.linspace(0, 2 * np.pi, len(D_A), endpoint=False) + 0.4


def stats(d):
    return d.mean(), (d < 1).mean(), (d < 5).mean()


def scatter_panel(ax, d, title, lim):
    """Predictions as offsets from the GT correspondence at the origin.

    Errors past the panel edge are clamped to the rim and labelled, so the
    sub-5-px structure stays readable next to a 45 px blowout.
    """
    ax.add_artist(plt.Circle((0, 0), 5, color=C_HIT5, alpha=0.10, zorder=0))
    ax.add_artist(plt.Circle((0, 0), 1, color=C_HIT1, alpha=0.18, zorder=0))
    for r, lab, dy in [(1, "τ = 1 px", -11), (5, "τ = 5 px", 4)]:
        ax.add_artist(plt.Circle((0, 0), r, fill=False, color=INK2, lw=1,
                                 ls=(0, (4, 3)), zorder=2))
        ax.annotate(lab, (-r * 0.71, -r * 0.71), xytext=(-4, dy), ha="right",
                    textcoords="offset points", fontsize=7.5, color=INK2,
                    zorder=4)
    rim = lim * 0.92
    r_plot = np.minimum(d, rim)
    x, y = r_plot * np.cos(ANG), r_plot * np.sin(ANG)
    colors = [C_HIT1 if v < 1 else C_HIT5 if v < 5 else C_MISS for v in d]
    for xi, yi, v in zip(x, y, d):
        if v > rim:
            ax.annotate(f"{v:.0f} px", (xi, yi), xytext=(0, 7), ha="center",
                        textcoords="offset points", fontsize=7.5, color=C_MISS)
    ax.scatter(x, y, s=42, c=colors, zorder=3, edgecolors=SURFACE, linewidths=1.2)
    ax.scatter([0], [0], marker="+", s=90, c=INK, zorder=5, linewidths=1.6)
    ax.annotate("GT match", (0, 0), xytext=(4, -11), textcoords="offset points",
                fontsize=7.5, color=INK)
    e, p1, p5 = stats(d)
    ax.set_title(f"{title}\nEPE {e:.2f} px  ·  PCK@1 {p1:.2f}  ·  PCK@5 {p5:.2f}",
                 fontsize=9.5, color=INK, loc="left")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("pixels from the true correspondence")


def main():
    fig = plt.figure(figsize=(11.4, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.28)
    ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))

    scatter_panel(ax0, D_A, "model A — accurate, 2 blowouts", 11)
    scatter_panel(ax1, D_B, "model B — uniformly mediocre", 11)

    # PCK is the CDF of the per-pixel error, read at tau.
    taus = np.linspace(0, 12, 400)
    for d, c, lab in [(D_A, C_A, "model A"), (D_B, C_B, "model B")]:
        ax2.plot(taus, [(d < t).mean() for t in taus], color=c, lw=2, label=lab)
        e = d.mean()
        ax2.axvline(e, color=c, lw=1, ls=(0, (2, 2)), alpha=0.7)
        ax2.annotate(f"EPE {e:.2f}", (e, 0.04), xytext=(3, 0), rotation=90,
                     textcoords="offset points", fontsize=7.5, color=c)
    for t in (1, 5):
        ax2.axvline(t, color=INK2, lw=1, ls=(0, (4, 3)))
        ax2.annotate(f"τ={t}", (t, 1.02), ha="center", fontsize=7.5, color=INK2)
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 1.08)
    ax2.set_xlabel("threshold τ (px)")
    ax2.set_ylabel("fraction of pixels within τ")
    ax2.set_title("PCK@τ is the CDF of the per-pixel error;\n"
                  "EPE is its mean — a different summary",
                  fontsize=9.5, color=INK, loc="left")
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle("EPE vs PCK: model B has the better EPE, model A matches "
                 "far more pixels correctly", fontsize=10.5, x=0.01, ha="left")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.14)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    for lab, d in [("A", D_A), ("B", D_B)]:
        e, p1, p5 = stats(d)
        print(f"model {lab}: EPE {e:.3f}  PCK@1 {p1:.3f}  PCK@5 {p5:.3f}")


if __name__ == "__main__":
    main()
