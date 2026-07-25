"""Matching losses: attention CE at /16, warp Charbonnier + conf BCE at /4.

Grid sizes are read off the tensors, so the same code serves any input
resolution. GT warps/covisibility come from CO3D depth+pose, never from the
prediction.
"""

import torch
import torch.nn.functional as F


def attn_ce_loss(logits, warp16, covis16):
    """CE of (B,Ha,Wa,Hb,Wb) logits vs bilinear soft GT cells; covis only.

    RoMa-style regression-by-classification: the GT match position is soft-
    binned bilinearly over its 4 neighbouring cells.
    """
    B, g = logits.shape[0], logits.shape[1]
    logp = F.log_softmax(logits.reshape(B, g * g, g * g), dim=-1)

    # non-covis cells can carry non-finite warps (fp16 depth overflow);
    # they are masked below, but nan.long() indexes out of bounds first
    warp16 = torch.nan_to_num(warp16, nan=0.0, posinf=0.0, neginf=0.0)
    c = ((warp16 + 1) * g / 2 - 0.5).clamp(0, g - 1)  # (B,g,g,2) cell coords
    f0 = c.floor().clamp(max=g - 2)
    w1 = (c - f0).clamp(0, 1)
    x0, y0 = f0[..., 0].long(), f0[..., 1].long()
    wx1, wy1 = w1[..., 0], w1[..., 1]
    loss = 0.0

    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        w = ((wx1 if dx else 1 - wx1) * (wy1 if dy else 1 - wy1)).reshape(B, -1)
        idx = ((y0 + dy) * g + (x0 + dx)).reshape(B, -1)
        loss = loss - w * logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    m = covis16.reshape(B, -1).float()
    return (loss * m).sum() / m.sum().clamp(min=1)


def warp_conf_loss(warp_pred, conf_pred, warp4, covis4):
    m = covis4.float()
    # zero (not multiply-mask) non-covis residuals: 0 * inf = nan
    warp4 = torch.nan_to_num(warp4, nan=0.0, posinf=0.0, neginf=0.0)
    charb = torch.sqrt(((warp_pred - warp4) ** 2).sum(-1) + 1e-6)
    l_warp = torch.where(covis4, charb, 0.0).sum() / m.sum().clamp(min=1)
    l_conf = F.binary_cross_entropy_with_logits(conf_pred[..., 0], m)
    return l_warp, l_conf


def matching_loss(preds, batch):
    """(attn CE, warp Charbonnier, confidence BCE), averaged over directions."""
    l_attn = attn_ce_loss(
        preds["attn_AB_logits"], batch["warp16_ab"], batch["covis16_ab"]
    ) + attn_ce_loss(preds["attn_BA_logits"], batch["warp16_ba"], batch["covis16_ba"])
    l_w_ab, l_c_ab = warp_conf_loss(
        preds["warp_AB"], preds["confidence_AB"], batch["warp4_ab"], batch["covis4_ab"]
    )
    l_w_ba, l_c_ba = warp_conf_loss(
        preds["warp_BA"], preds["confidence_BA"], batch["warp4_ba"], batch["covis4_ba"]
    )
    return l_attn / 2, (l_w_ab + l_w_ba) / 2, (l_c_ab + l_c_ba) / 2


def recon_loss(preds, batch, lpips_fn, lpips_w=1.0):
    """L1 + lpips_w * LPIPS(VGG), averaged over the two views."""
    xr_A, xr_B = preds["recon_A"], preds["recon_B"]
    l1 = (
        (xr_A - batch["img_A"]).abs().mean() + (xr_B - batch["img_B"]).abs().mean()
    ) / 2
    lp = (
        lpips_fn(batch["img_A"] * 2 - 1, xr_A * 2 - 1).mean()
        + lpips_fn(batch["img_B"] * 2 - 1, xr_B * 2 - 1).mean()
    ) / 2
    return l1, lp, l1 + lpips_w * lp
