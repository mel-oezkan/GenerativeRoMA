"""The released RoMaV2 (descriptor + matcher + refiners), wrapped for our evals.

Every wrapper here forces amp off — each eval in this repo runs fp32 (Pascal
has no native bf16) — so the only difference left between the `pretrained`
arm and these is the piece being measured.
"""

import sys
from dataclasses import replace

import torch
import torch.nn.functional as F

from src.paths import ROMAV2_SRC
from src.r2.model import DESC_CH, DescProj, load_run_ckpt, matcher_state_dict

if str(ROMAV2_SRC) not in sys.path:
    sys.path.insert(0, str(ROMAV2_SRC))


def _configure(model):
    """amp off on both the matcher and its DPT head."""
    model.matcher.cfg = replace(model.matcher.cfg, enable_amp=False)
    model.matcher.head.enable_amp = False
    return model


def _released_romav2(cls=None):
    from romav2.romav2 import RoMaV2

    torch.set_float32_matmul_precision("highest")  # required by its forward
    cls = cls or RoMaV2
    return _configure(cls(RoMaV2.Cfg(setting="turbo", compile=False)).eval())


class RefinedPretrained(torch.nn.Module):
    """Full released model adapted to the R2Model eval signature.

    The refiners output a warp at full resolution, resampled here onto the
    stride-4 grid the GT warps live on (bilinear to 80x80 at 320 px = the
    centres of the 4x4 blocks). The pre-refiner warp is passed through as
    well, so one run reports both the like-for-like coarse-only number and
    the refined one.
    """

    def __init__(self, grid4=80):
        super().__init__()
        self.model = _released_romav2()
        self.grid4 = grid4

    def forward(self, desc_A, desc_B, img_A, img_B):  # desc unused: recomputed
        out = self.model(img_A, img_B)
        warp = F.interpolate(
            out["warp_AB"].permute(0, 3, 1, 2).float(),
            size=(self.grid4, self.grid4), mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)
        return {
            "warp_AB": warp,
            "warp_AB_matcher": out["matcher"]["warp_AB"].float(),
            "attn_AB_logits": out["matcher"]["attn_AB_logits"],
        }


def _coarse_class():
    """RoMaV2 whose match() works without refiners.

    The released match() maps `confidence` through _map_confidence, which
    reads channels 1:4 as precision parameters — those only exist on the
    ConvRefiners' output. The matcher stage emits a single overlap logit, so
    we build the preds dict directly and hand sample() an identity precision
    (mega1500/scannet1500 discard the precisions sample returns, so this is
    inert; it just keeps the non-bidirectional branch, which dereferences
    precision unconditionally, from crashing).
    """
    from romav2.romav2 import RoMaV2

    class CoarseRoMaV2(RoMaV2):
        def match(self, img_like_A, img_like_B):
            self.eval()

            def lr(x):
                return F.interpolate(x, size=(self.H_lr, self.W_lr),
                                     mode="bicubic", align_corners=False,
                                     antialias=True)

            preds = self(lr(self._load_image(img_like_A)),
                         lr(self._load_image(img_like_B)))

            # The matcher emits a /4 warp (80x80 at 320 px). sample() draws
            # 4 x 5000 correspondences without replacement, which only fits
            # because the refiners hand it a full-resolution warp -- so
            # upsample to H_lr x W_lr here. That also equalises the sampling
            # budget between the refiner-free arms and the released anchor.
            def up(x):
                return F.interpolate(
                    x.permute(0, 3, 1, 2).float(), size=(self.H_lr, self.W_lr),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1).contiguous()

            warp, conf = up(preds["warp_AB"]), up(preds["confidence_AB"])
            B, H, W, _ = warp.shape
            eye = torch.eye(2, device=warp.device).expand(B, H, W, 2, 2)
            return {
                "warp_AB": warp.clone(),
                "confidence_AB": conf.clone(),
                "overlap_AB": conf[..., :1].sigmoid().clone(),
                "precision_AB": eye.contiguous(),
                "warp_BA": None, "confidence_BA": None,
                "overlap_BA": None, "precision_BA": None,
            }

    return CoarseRoMaV2


def benchmark_model(run, arm, step, desc_only=False, refiners=False):
    """Released RoMaV2 with our matcher weights swapped in, for the pose
    benchmarks (which need its match/sample/pose extraction).

    `setting="turbo"` -> 320 px, no hr pass, the resolution our matchers were
    trained at (mega1500's native 800/1024 would confound a resolution shift
    into the transfer result). The refiners are dropped unless asked for,
    since no arm of ours has them; forward() then returns the coarse /4 warp.
    """
    model = _released_romav2(None if refiners else _coarse_class())
    if desc_only:
        model.matcher.mv_vit = DescProj(DESC_CH, model.matcher.cfg.dim).to(
            next(model.parameters()).device
        )
    if arm != "pretrained":
        ckpt = load_run_ckpt(run, arm, step)
        model.matcher.load_state_dict(matcher_state_dict(ckpt["model"]), strict=True)
    if not refiners:
        model.refiners = torch.nn.ModuleDict()  # forward() -> coarse warp
    return model.eval().requires_grad_(False)


def eval_model(run, arm, step, desc_only=False, grid4=80, device="cuda"):
    """Model for the per-category eval: a trained arm, the stock matcher
    (`pretrained`), or the full released system (`pretrained-refined`)."""
    from src.r2.model import load_trained_model

    if arm == "pretrained-refined":
        return RefinedPretrained(grid4).to(device).eval().requires_grad_(False)
    return load_trained_model(run, arm, step, desc_only=desc_only, device=device)
