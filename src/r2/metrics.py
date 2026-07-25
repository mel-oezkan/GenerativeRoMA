"""The matching eval, in one place.

EPE/PCK of the /4 warp plus a coarse argmax EPE at /16, AB direction,
covisible pixels only. Rows carry the CO3D category and a near/far tag
(|i - j| of the pair key), so the same rows aggregate into the training-time
summary and into the per-category breakdown of r3_eval_cats.
"""

from pathlib import Path

import numpy as np
import torch

from src.r2.dataset import as_batch

SCOPES = ["all", "near", "far"]
METRIC_KEYS = ["epe", "pck1", "pck3", "pck5", "epe_coarse"]
# only produced by the refined-pretrained arm (pre-refiner control column)
EXTRA_KEYS = ["epe_matcher", "pck5_matcher"]
MIN_COVIS = 16  # pairs with less GT than this carry no measurement


@torch.no_grad()
def eval_rows(model, eval_ds, size=320, device="cuda", max_pairs=None):
    """One row per usable eval pair."""
    was_training = model.training
    model.eval()
    rows = []
    n = len(eval_ds) if max_pairs is None else min(max_pairs, len(eval_ds))
    for i in range(n):
        b = eval_ds[i]
        if b["covis4_ab"].sum() < MIN_COVIS:
            continue
        batch = as_batch(b, device)
        preds = model(batch["desc_A"], batch["desc_B"], batch["img_A"], batch["img_B"])

        m4 = batch["covis4_ab"][0]
        epe4 = ((preds["warp_AB"][0] - batch["warp4_ab"][0]) / 2 * size).norm(
            dim=-1
        )[m4]
        m16 = batch["covis16_ab"][0]
        g = preds["attn_AB_logits"].shape[1]
        am = preds["attn_AB_logits"][0].reshape(g * g, g * g).argmax(-1)
        cell = torch.stack([am % g, am // g], -1).reshape(g, g, 2).float()
        coarse = (cell + 0.5) * (2 / g) - 1
        epe16 = ((coarse - batch["warp16_ab"][0]) / 2 * size).norm(dim=-1)[m16]

        ij = b["key"].split("_")[-2:]
        row = {
            "cat": Path(eval_ds.pairs[i]["anchor"]).parts[-4],
            "near": abs(int(ij[0]) - int(ij[1])) <= 4,
            "epe": epe4.mean().item(),
            "pck1": (epe4 < 1).float().mean().item(),
            "pck3": (epe4 < 3).float().mean().item(),
            "pck5": (epe4 < 5).float().mean().item(),
            "epe_coarse": epe16.mean().item() if m16.any() else float("nan"),
        }
        if "warp_AB_matcher" in preds:  # refined arm: pre-refiner control
            epe_m = (
                (preds["warp_AB_matcher"][0] - batch["warp4_ab"][0]) / 2 * size
            ).norm(dim=-1)[m4]
            row["epe_matcher"] = epe_m.mean().item()
            row["pck5_matcher"] = (epe_m < 5).float().mean().item()
        rows.append(row)
    model.train(was_training)
    return rows


def aggregate(rows, per_cat=False):
    """Rows -> {all,near,far}[+per_cat] means. Empty scopes are omitted."""
    if not rows:
        return {}
    keys = METRIC_KEYS + [k for k in EXTRA_KEYS if k in rows[0]]

    def agg(rs):
        out = {k: float(np.nanmean([r[k] for r in rs])) for k in keys}
        out["n"] = len(rs)
        return out

    def by_scope(rs):
        out = {}
        for scope in SCOPES:
            sub = [r for r in rs if scope == "all" or r["near"] == (scope == "near")]
            if sub:
                out[scope] = agg(sub)
        return out

    out = by_scope(rows)
    if per_cat:
        out["per_cat"] = {
            cat: by_scope([r for r in rows if r["cat"] == cat])
            for cat in sorted({r["cat"] for r in rows})
        }
    return out


def evaluate(model, eval_ds, size=320, device="cuda", max_pairs=None,
             per_cat=False):
    """eval_rows + aggregate — the training loop's one-liner."""
    return aggregate(
        eval_rows(model, eval_ds, size, device, max_pairs), per_cat=per_cat
    )


def per_category_eval(cfg):
    """Re-run the eval on a trained arm, broken down by CO3D category.

    Writes results/<run>/<arm>/metrics_per_cat[_<tag>].json. `arm=pretrained`
    and `arm=pretrained-refined` evaluate the released model instead of a run.
    """
    import json

    from src.co3d_geom import load_frame_index_multi
    from src.paths import RESULTS_DIR
    from src.r2.dataset import R2PairDataset
    from src.r2.released import eval_model
    from src.splits import categories_for, get_splits

    index = load_frame_index_multi(categories_for(cfg.split))
    _, eval_pairs = get_splits(cfg.split)
    eval_ds = R2PairDataset(eval_pairs, index, cfg.size)
    model = eval_model(cfg.run, cfg.arm, cfg.step, cfg.model.desc_only,
                       cfg.size // 4, cfg.device)

    out = aggregate(eval_rows(model, eval_ds, cfg.size, cfg.device), per_cat=True)
    out_dir = RESULTS_DIR / cfg.run / cfg.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"metrics_per_cat{'_' + cfg.tag if cfg.tag else ''}.json"
    (out_dir / name).write_text(json.dumps(out, indent=1))

    print(f"{cfg.run}/{cfg.arm} [{cfg.split}]: {out['all']['n']} pairs")
    for cat, m in out["per_cat"].items():
        print(
            f"  {cat:10s} EPE {m['all']['epe']:6.2f} "
            f"(near {m['near']['epe']:5.2f} / far {m['far']['epe']:6.2f})  "
            f"PCK5 {m['all']['pck5']:.3f}  n={m['all']['n']}"
        )
    return out
