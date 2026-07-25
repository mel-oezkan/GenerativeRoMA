"""Qualitative panels written next to each R2/R3 run's metrics."""

import torch
import torch.nn.functional as F

from src.r2.dataset import as_batch
from src.viz.style import agg_figure


def save_panels(model, eval_ds, out_dir, size=320, device="cuda", n=4):
    """Rows of (A, B, B warped by the predicted warp, predicted confidence)."""
    plt, fig, axes = agg_figure(n, 4, figsize=(13, 3.3 * n))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        shown = 0
        for i in range(len(eval_ds)):
            if shown >= n:
                break
            b = eval_ds[i]
            if b["covis4_ab"].sum() < 16:
                continue
            batch = as_batch(b, device)
            preds = model(
                batch["desc_A"], batch["desc_B"], batch["img_A"], batch["img_B"]
            )
            conf = preds["confidence_AB"][0, ..., 0].sigmoid()
            warped = F.grid_sample(
                batch["img_B"],
                preds["warp_AB"][0][None].clamp(-1, 1),
                mode="bilinear",
                align_corners=False,
            )[0]
            warped = F.interpolate(warped[None], (size, size), mode="bilinear")[0]
            conf_up = F.interpolate(conf[None, None], (size, size), mode="bilinear")[
                0, 0
            ]
            panels = [
                (b["img_A"].permute(1, 2, 0), "A"),
                (b["img_B"].permute(1, 2, 0), "B"),
                (
                    (warped * (conf_up > 0.5)).permute(1, 2, 0).cpu(),
                    "B warped by pred (conf>.5)",
                ),
                (conf_up.cpu(), "pred confidence"),
            ]
            for ax, (im, t) in zip(axes[shown], panels):
                ax.imshow(
                    im.numpy().clip(0, 1) if im.dim() == 3 else im.numpy(),
                    cmap=None if im.dim() == 3 else "viridis",
                )
                ax.set_title(f"{t} [{b['key'][:24]}]" if t == "A" else t, fontsize=7)
                ax.axis("off")
            shown += 1
    fig.tight_layout()
    fig.savefig(out_dir / "warp_panels.png", dpi=110)
    plt.close(fig)
    model.train(was_training)


def save_recon_grid(model, eval_ds, out_dir, device="cuda", n=6):
    """Target row + train-time-decoder reconstruction row."""
    plt, fig, axes = agg_figure(2, n, figsize=(2.2 * n, 4.6))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for col in range(n):
            b = eval_ds[col * (len(eval_ds) // n)]
            batch = as_batch(b, device)
            preds = model(
                batch["desc_A"], batch["desc_B"], batch["img_A"], batch["img_B"]
            )
            axes[0, col].imshow(b["img_A"].permute(1, 2, 0).numpy())
            axes[1, col].imshow(
                preds["recon_A"][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            )
            for r in range(2):
                axes[r, col].axis("off")
    axes[0, 0].set_title("target", loc="left")
    axes[1, 0].set_title("recon (train-time decoder)", loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "recon_grid.png", dpi=120)
    plt.close(fig)
    model.train(was_training)
