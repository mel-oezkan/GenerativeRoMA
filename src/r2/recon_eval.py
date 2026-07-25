"""Held-out recon eval of a run's train-time decoder (docs/R2.md).

Same metric trio as the R1 probe (PSNR/SSIM/LPIPS-alex), but for the decoder
that was co-trained with its encoder rather than the standardized post-hoc
probe decoder — the fair comparison between equal-budget arms.

View A is the decoder's train target in every arm; view B is a train target
only for the two-view run (v2). For view-A-only runs the B column is a
zero-shot read of whether B's mv tokens decode at all.
"""

import json

import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

from src.co3d_geom import load_frame_index_multi
from src.paths import RESULTS_DIR
from src.r2.dataset import R2PairDataset, as_batch, build_train_pairs
from src.r2.model import load_trained_model
from src.splits import categories_for


def image_metrics(xr, x, lpips_fn):
    xr = xr.clamp(0, 1)
    mse = ((xr - x) ** 2).mean()
    return {
        "psnr": (-10 * torch.log10(mse)).item(),
        "ssim": ssim_fn(xr, x, data_range=1.0).item(),
        "lpips": lpips_fn(xr * 2 - 1, x * 2 - 1).mean().item(),
    }


@torch.no_grad()
def recon_eval(cfg):
    import lpips

    index = load_frame_index_multi(categories_for(cfg.split))
    _, eval_pairs = build_train_pairs(index, cfg.split, cfg.data.vq_min,
                                      verbose=False)
    eval_ds = R2PairDataset(eval_pairs, index, cfg.size)
    model = load_trained_model(
        cfg.run, cfg.arm, cfg.step, with_decoder=True,
        desc_only=cfg.model.desc_only, skip_head=cfg.model.skip_head,
        device=cfg.device,
    )
    lpips_fn = lpips.LPIPS(net="alex").to(cfg.device).eval()

    rows = {"A": [], "B": []}
    for i in range(len(eval_ds)):
        batch = as_batch(eval_ds[i], cfg.device)
        preds = model(batch["desc_A"], batch["desc_B"], batch["img_A"],
                      batch["img_B"])
        for view in ("A", "B"):
            rows[view].append(
                image_metrics(preds[f"recon_{view}"], batch[f"img_{view}"], lpips_fn)
            )

    out = {"n": len(rows["A"])}
    for view in ("A", "B"):
        out[view] = {k: sum(r[k] for r in rows[view]) / len(rows[view])
                     for k in rows[view][0]}
    out["mean"] = {k: (out["A"][k] + out["B"][k]) / 2 for k in out["A"]}
    out_path = RESULTS_DIR / cfg.run / cfg.arm / "recon_eval.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"{cfg.run}/{cfg.arm}: {json.dumps(out)}")
    return out
