"""Report figures for docs/R3.md (and the R2 v1 section of docs/R2.md).

Reads the metrics JSONs produced by r2_train.py, r3_eval_cats.py and the
post-hoc R1 probes, writes PNGs under docs/figures/. Figures whose inputs are
missing are skipped, so the script can be rerun as results land.

    python experiments/visualizations/r3_report_figs.py
    python experiments/visualizations/r3_report_figs.py "figures=[probe,tradeoff]"

Configs: configs/figures/report.yaml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from src.config import hydra_main  # noqa: E402
from src.viz.io import fig_dir, load_result as load  # noqa: E402
from src.viz.style import (  # noqa: E402
    BASE,
    C_JOINT,
    C_MATCH,
    C_PRE,
    C_RECON,
    GRID,
    INK,
    INK2,
    MUTED,
    SURFACE,
    bar_labels,
    grouped,
    pyplot,
)

plt = pyplot()

# Filled in by main() from configs/figures/report.yaml: the output directory
# and the R1 reference measurements the figures annotate against.
OUT = None
R1_PRE_MV = R1_DESC = None


def fig_r3_matching():
    pre = load("r3_ft/pretrained/metrics_per_cat.json")
    m = load("r3_ft/match/metrics.json")
    j = load("r3_ft/joint/metrics.json")
    if not (pre and m and j):
        return print("skip fig_r3_matching (missing inputs)")
    scopes = ["all", "near", "far"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    series = [
        ("pretrained (zero-shot)", C_PRE, pre),
        ("ft-match", C_MATCH, m),
        ("ft-joint (+recon)", C_JOINT, j),
    ]
    for ax, key, fmt, title in [
        (axes[0], "epe", "{:.2f}", "EPE at 320 px (lower is better)"),
        (axes[1], "pck5", "{:.3f}", "PCK@5 px (higher is better)"),
    ]:
        rs = grouped(
            ax, scopes, [(lab, c, [d[s][key] for s in scopes]) for lab, c, d in series]
        )
        for r in rs:
            bar_labels(ax, r, fmt)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.margins(y=0.18)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "R3: fine-tuning the pretrained RoMaV2 matcher on CO3D (209 held-out pairs, 4 categories)",
        fontsize=10.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig_r3_matching.png", dpi=150)
    plt.close(fig)
    print("wrote fig_r3_matching.png")


def fig_per_category():
    pre = load("r3_ft/pretrained/metrics_per_cat.json")
    m = load("r3_ft/match/metrics_per_cat.json")
    j = load("r3_ft/joint/metrics_per_cat.json")
    if not (pre and m and j):
        return print("skip fig_per_category (missing inputs)")
    cats = sorted(m["per_cat"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    series = [
        ("pretrained (zero-shot)", C_PRE, pre),
        ("ft-match", C_MATCH, m),
        ("ft-joint (+recon)", C_JOINT, j),
    ]
    for ax, key, fmt, title in [
        (axes[0], "epe", "{:.1f}", "EPE by category (lower is better)"),
        (axes[1], "pck5", "{:.2f}", "PCK@5 px by category (higher is better)"),
    ]:
        rs = grouped(
            ax,
            cats,
            [
                (lab, c, [d["per_cat"][cat]["all"][key] for cat in cats])
                for lab, c, d in series
            ],
        )
        for r in rs:
            bar_labels(ax, r, fmt)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.margins(y=0.18)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_category.png", dpi=150)
    plt.close(fig)
    print("wrote fig_per_category.png")


def fig_probe_psnr():
    rows = [("pretrained (romav2.pt)", C_PRE, R1_PRE_MV["psnr"])]
    for run, fam, scale in [
        ("r2_probe", "scratch", "6k"),
        ("r2_v1_probe", "scratch", "12k"),
        ("r3_probe", "ft", "6k"),
    ]:
        for arm, color in [("match", C_MATCH), ("joint", C_JOINT)]:
            d = load(f"{run}/{arm}/metrics.json")
            if d:
                # only the 12k scratch joint arm trained with the delayed GAN
                gan = " +GAN" if (run, arm) == ("r2_v1_probe", "joint") else ""
                rows.append((f"{fam}-{arm} {scale}{gan}", color, d["psnr"]))
    if len(rows) < 3:
        return print("skip fig_probe_psnr (missing inputs)")
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    ax.axvline(R1_DESC["psnr"], color=INK2, lw=1, ls=(0, (4, 3)))
    for yi, (lab, color, v) in zip(y, rows):
        ax.barh(yi, v, 0.62, color=color)
        ax.annotate(
            f"{v:.2f}", (v, yi), ha="left", va="center", fontsize=8,
            color=INK2, xytext=(3, 0), textcoords="offset points",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.5),
        )
    ax.set_ylim(-0.6, len(rows) - 0.4 + 0.9)
    ax.annotate(
        f"frozen-desc ceiling {R1_DESC['psnr']:.2f} dB",
        (R1_DESC["psnr"], len(rows) - 0.4 + 0.45),
        ha="right", va="center", fontsize=8, color=INK2, xytext=(-5, 0),
        textcoords="offset points",
    )
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(length=0)
    ax.set_xlim(0, 21.5)
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=c, label=l)
            for c, l in [
                (C_MATCH, "matching only"),
                (C_JOINT, "matching + recon"),
                (C_PRE, "reference"),
            ]
        ],
        loc="upper left", ncols=3, fontsize=8, columnspacing=1.2,
        handlelength=1.2, borderaxespad=0.1,
    )
    ax.set_title(
        "Appearance retained in the mv tokens\n"
        "post-hoc RAE-probe PSNR (dB), hydrant-full protocol",
        fontsize=10, color=INK, loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_probe_psnr.png", dpi=150)
    plt.close(fig)
    print("wrote fig_probe_psnr.png")


def fig_tradeoff():
    """Matching (far-pair EPE, r3-4cat eval) vs appearance (probe PSNR)."""
    pts = []  # (label, color, x, y, marker, label offset)
    off = (6, 5)
    pre = load("r3_ft/pretrained/metrics_per_cat.json")
    if pre:
        pts.append(("pretrained", C_PRE, R1_PRE_MV["psnr"], pre["far"]["epe"], "o", off))
    for run, probe, fam, scale, mk in [
        ("r3_ft", "r3_probe", "ft", "6k", "o"),
        ("r2_v1", "r2_v1_probe", "scratch", "12k", "s"),
    ]:
        for arm, color in [("match", C_MATCH), ("joint", C_JOINT)]:
            met = load(f"{run}/{arm}/metrics.json")
            prb = load(f"{probe}/{arm}/metrics.json")
            if met and prb:
                pts.append((f"{fam}-{arm} {scale}", color, prb["psnr"],
                            met["far"]["epe"], mk, off))
    # frozen-desc baselines: no mv tokens, probe PSNR = desc ceiling by construction
    for arm, color, aoff in [
        ("match", C_MATCH, (8, -12)),
        ("joint", C_JOINT, (8, 4)),
    ]:
        met = load(f"r2_v1_desc/{arm}/metrics.json")
        if met:
            pts.append((f"desc-{arm} 12k", color, R1_DESC["psnr"],
                        met["far"]["epe"], "^", aoff))
    if len(pts) < 4:
        return print("skip fig_tradeoff (missing probe results)")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lab, color, x, yv, mk, aoff in pts:
        ax.scatter(x, yv, s=70, color=color, marker=mk, zorder=3,
                   edgecolors=SURFACE, linewidths=2)
        ax.annotate(lab, (x, yv), xytext=aoff, textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.axvline(R1_DESC["psnr"], color=INK2, lw=1, ls=(0, (4, 3)))
    ax.margins(x=0.12)
    ax.annotate("desc ceiling", (R1_DESC["psnr"], ax.get_ylim()[0]),
                ha="right", va="bottom", fontsize=8, color=INK2, xytext=(-4, 4),
                textcoords="offset points")
    ax.set_xlabel("appearance: probe PSNR of mv tokens (dB) →")
    ax.set_ylabel("← matching: far-pair EPE (px, r3-4cat eval)")
    ax.set_title(
        "The information/reconstruction trade-off, per model\n"
        "(up-left = collapsed, down-right = matches well and keeps appearance)",
        fontsize=10, color=INK, loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_tradeoff.png", dpi=150)
    plt.close(fig)
    print("wrote fig_tradeoff.png")


def view_a(d):
    """recon_eval.json in either format -> the view-A metrics.

    The two-view eval (r2_recon_eval.py, v2 onwards) nests per-view dicts
    under A/B/mean; older runs wrote the view-A numbers flat. View A is the
    decoder's train target in every arm, so it is the comparable column.
    """
    return d["A"] if "A" in d else d


def missing_labels(ax, x, vals, text):
    """Mark the arms an axis does not apply to (zero-height bars)."""
    for xi, v in zip(x, vals):
        if not v:
            ax.annotate(text, (xi, 0), xytext=(0, 6), rotation=90,
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=MUTED)


def fig_recon_quality():
    """Train-time recon eval + matching of the equal-budget 12k+GAN runs."""
    # colour = arm (match / joint / recon), x position groups the two encoder
    # families; match-only arms have no decoder, recon-only has no matcher.
    rows = [
        ("deep\nmatch", C_MATCH, "r2_v1/match"),
        ("deep\njoint 1v", C_JOINT, "r2_v1/joint"),
        ("deep\njoint 2v", C_JOINT, "r2_v2/joint"),
        ("DINOv3\nmatch", C_MATCH, "r2_v1_desc/match"),
        ("DINOv3\njoint", C_JOINT, "r2_v1_desc/joint"),
        ("DINOv3\nrecon", C_RECON, "r2_v1_desc/recon"),
    ]
    data = [
        (lab, c, load(f"{run}/recon_eval.json"), load(f"{run}/metrics.json"))
        for lab, c, run in rows
    ]
    if all(d is None for _, _, d, _ in data):
        return print("skip fig_recon_quality (missing recon_eval.json)")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.0))
    for ax, key, fmt, title in [
        (axes[0], "psnr", "{:.2f}", "recon PSNR (dB, higher is better)"),
        (axes[1], "lpips", "{:.3f}", "recon LPIPS (lower is better)"),
    ]:
        x = np.arange(len(data))
        vals = [view_a(d)[key] if d else 0.0 for _, _, d, _ in data]
        r = ax.bar(x, vals, 0.55, color=[c for _, c, _, _ in data])
        bar_labels(ax, [b for b, v in zip(r, vals) if v], fmt)
        missing_labels(ax, x, vals, "no decoder")
        ax.set_xticks(x, [lab for lab, _, _, _ in data], fontsize=8)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.margins(y=0.18)
    # matching panel: far-pair EPE of the same checkpoints (recon-only arm
    # never runs the matching pipeline, so it has no bar).
    ax = axes[2]
    x = np.arange(len(data))
    epe = [m["far"]["epe"] if m and "far" in m else 0.0 for _, _, _, m in data]
    r = ax.bar(x, epe, 0.55, color=[c for _, c, _, _ in data])
    bar_labels(ax, [b for b, v in zip(r, epe) if v], "{:.2f}")
    missing_labels(ax, x, epe, "no matcher")
    ax.set_xticks(x, [lab for lab, _, _, _ in data], fontsize=8)
    ax.tick_params(length=0)
    ax.set_title("far-pair EPE (px, lower is better)", fontsize=9.5,
                 color=INK, loc="left")
    ax.margins(y=0.18)
    # reference only, not equal-budget arms: the released romav2.pt run
    # zero-shot on the same 209 pairs (trained on far more data, higher res).
    # Our arms have no refiners, so the coarse-only row is the like-for-like
    # one and the refined row is the headroom the ConvRefiners buy.
    refs = [  # (path, label template, label offset: above / below the line)
        ("r3_ft/pretrained/metrics_per_cat.json",
         "pretrained romav2.pt, no refiners ({:.2f})", 4),
        ("r3_ft/pretrained-refined/metrics_per_cat.json",
         "+ refiners ({:.2f})", -11),
    ]
    for path, tmpl, dy in refs:
        d = load(path)
        if not d:
            continue
        v = d["far"]["epe"]
        ax.axhline(v, color=C_PRE, lw=1.2, ls=(0, (4, 3)), zorder=4)
        ax.annotate(tmpl.format(v), (-0.45, v), xytext=(0, dy), ha="left",
                    textcoords="offset points", fontsize=7.5, color=INK2,
                    zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    fig.suptitle(
        "Equal-budget 12k+GAN runs: recon quality of the train-time decoders "
        "vs matching accuracy (r3-4cat eval, 237 / 209 pairs)",
        fontsize=10, x=0.02, ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig_recon_quality.png", dpi=150)
    plt.close(fig)
    print("wrote fig_recon_quality.png")


def fig_scaling():
    """scratch 6k (v0, hydrant) -> 12k (v1, 4-cat): probe + matching trends.

    Not a controlled steps sweep — steps, data breadth and the joint GAN
    all scale together (stated in the doc text); the matching panel uses
    the shared r3-4cat eval for both scales (v0 via r3_eval_cats.py).
    """
    probe = {
        arm: [load(f"{run}/{arm}/metrics.json") for run in ("r2_probe", "r2_v1_probe")]
        for arm in ("match", "joint")
    }
    if any(d is None for v in probe.values() for d in v):
        return print("skip fig_scaling (missing probe results)")
    match_cat = [load(f"{run}/match/metrics_per_cat.json") for run in ("r2_v0", "r2_v1")]
    joint_cat = [load(f"{run}/joint/metrics_per_cat.json") for run in ("r2_v0", "r2_v1")]
    have_matching = all(match_cat) and all(joint_cat)

    x = [0, 1]
    ticks = ["scratch 6k\n(hydrant only)", "scratch 12k\n(4-cat, joint +GAN)"]
    fig, axes = plt.subplots(1, 2 if have_matching else 1,
                             figsize=(8.6 if have_matching else 4.6, 3.4),
                             squeeze=False)
    ax = axes[0, 0]
    ax.axhline(R1_DESC["psnr"], color=INK2, lw=1, ls=(0, (4, 3)))
    ax.annotate("desc ceiling", (0.5, R1_DESC["psnr"]), ha="center",
                xytext=(0, 3), textcoords="offset points", fontsize=8,
                color=INK2)
    ax.axhline(R1_PRE_MV["psnr"], color=MUTED, lw=1, ls=(0, (1, 2)))
    ax.annotate("pretrained matcher (collapsed)", (0.5, R1_PRE_MV["psnr"]),
                ha="center", xytext=(0, -10), textcoords="offset points",
                fontsize=8, color=MUTED)
    for arm, color, aoff in [("match", C_MATCH, (6, -11)), ("joint", C_JOINT, (6, 6))]:
        ys = [d["psnr"] for d in probe[arm]]
        ax.plot(x, ys, "-o", color=color, lw=2, ms=6, label=arm)
        for xi, yi in zip(x, ys):
            ax.annotate(f"{yi:.2f}", (xi, yi), xytext=aoff,
                        textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xticks(x, ticks, fontsize=8)
    ax.set_xlim(-0.25, 1.35)
    ax.set_ylim(12, 20.3)
    ax.tick_params(length=0)
    ax.set_title("Appearance: probe PSNR (dB)", fontsize=9.5, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8)

    if have_matching:
        ax = axes[0, 1]
        for arm, cats, color, aoff in [("match", match_cat, C_MATCH, (6, 7)),
                                       ("joint", joint_cat, C_JOINT, (6, -11))]:
            ys = [d["far"]["epe"] for d in cats]
            ax.plot(x, ys, "-o", color=color, lw=2, ms=6, label=f"{arm}, all 4 cats")
            yh = [d["per_cat"]["hydrant"]["far"]["epe"] for d in cats]
            ax.plot(x, yh, "--o", color=color, lw=1.2, ms=4, alpha=0.6,
                    label=f"{arm}, hydrant only")
            for xi, yi in zip(x, ys):
                ax.annotate(f"{yi:.1f}", (xi, yi), xytext=aoff,
                            textcoords="offset points", fontsize=8, color=INK2)
        ax.set_xticks(x, ticks, fontsize=8)
        ax.set_xlim(-0.25, 1.35)
        ax.tick_params(length=0)
        ax.set_title("Matching: far-pair EPE (px, r3-4cat eval)",
                     fontsize=9.5, color=INK, loc="left")
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scaling.png", dpi=150)
    plt.close(fig)
    print("wrote fig_scaling.png")


ARMS = [  # (label, colour, run/arm, in-domain metrics.json or per_cat)
    ("v1 match\n(scratch)", C_MATCH, "r2_v1/match"),
    ("v1 joint\n(scratch)", C_JOINT, "r2_v1/joint"),
    ("v2 joint\n(2-view)", C_JOINT, "r2_v2/joint"),
    ("desc match\n(DINOv3)", C_MATCH, "r2_v1_desc/match"),
    ("desc joint\n(DINOv3)", C_JOINT, "r2_v1_desc/joint"),
    ("ft match\n(pretr. init)", C_MATCH, "r3_ft/match"),
    ("ft joint\n(pretr. init)", C_JOINT, "r3_ft/joint"),
    ("pretrained\n(zero-shot)", C_PRE, "r3_ft/pretrained"),
]


def macro_epe(d, scope="all"):
    """Category-averaged EPE.

    The co3d-unseen split is unbalanced (apple/suitcase/toytrain contribute
    5-6 sequences from co3d_full, the other 14 categories 2 each from
    co3d_data), so the pooled mean is weighted toward three categories.
    Averaging the per-category means instead gives every category one vote.
    """
    cats = d.get("per_cat")
    if not cats:
        return None
    vals = [m[scope]["epe"] for m in cats.values() if m.get(scope, {}).get("n")]
    return float(np.mean(vals)) if vals else None


def fig_generalization():
    """In-domain vs held-out-category matching, same protocol and metrics."""
    data = []
    for lab, color, key in ARMS:
        run, arm = key.split("/")
        ind = load(f"{key}/metrics.json") or load(f"{key}/metrics_per_cat.json")
        out = load(f"{key}/metrics_per_cat_co3d-unseen.json")
        if ind and out:
            data.append((lab, color, ind, out))
    if len(data) < 4:
        return print("skip fig_generalization (missing co3d-unseen evals)")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4))
    x = np.arange(len(data))
    ax = axes[0]
    series = [
        ("r3-4cat (in domain)", [d[2]["all"]["epe"] for d in data], 1.0),
        ("co3d-unseen (17 held-out cats)", [d[3]["all"]["epe"] for d in data], 0.45),
    ]
    for i, (lab, vals, alpha) in enumerate(series):
        off = (i - 0.5) * 0.42
        r = ax.bar(x + off, vals, 0.40, label=lab, alpha=alpha,
                   color=[d[1] for d in data])
        bar_labels(ax, r, "{:.1f}")
    ax.set_xticks(x, [d[0] for d in data], fontsize=7.5)
    ax.tick_params(length=0)
    ax.margins(y=0.18)
    ax.set_title("EPE (px): in domain (solid) vs unseen categories (faded)",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, loc="upper right")

    # Right: the transfer gap itself, on the category-balanced average.
    ax = axes[1]
    macro = [macro_epe(d[3]) for d in data]
    r = ax.bar(x, macro, 0.55, color=[d[1] for d in data])
    bar_labels(ax, r, "{:.1f}")
    ax.set_xticks(x, [d[0] for d in data], fontsize=7.5)
    ax.tick_params(length=0)
    ax.margins(y=0.18)
    ax.set_title("co3d-unseen EPE (px), averaged per category (unweighted)",
                 fontsize=9.5, color=INK, loc="left")
    fig.suptitle(
        "Category transfer: trained on hydrant/bench/toybus/toytruck, "
        "evaluated on 17 disjoint CO3D categories (413 pairs)",
        fontsize=10, x=0.02, ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig_generalization.png", dpi=150)
    plt.close(fig)
    print("wrote fig_generalization.png")


@hydra_main("figures/report")
def main(cfg):
    global OUT, R1_PRE_MV, R1_DESC
    OUT = fig_dir(cfg.out_dir)
    R1_PRE_MV = cfg.reference.pretrained_mv
    R1_DESC = cfg.reference.desc_ceiling

    which = list(cfg.figures)
    fns = {
        "matching": fig_r3_matching,
        "cats": fig_per_category,
        "probe": fig_probe_psnr,
        "tradeoff": fig_tradeoff,
        "reconq": fig_recon_quality,
        "scaling": fig_scaling,
        "generalization": fig_generalization,
    }
    for name, fn in fns.items():
        if "all" in which or name in which:
            fn()


if __name__ == "__main__":
    main()
