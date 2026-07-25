"""The R2/R3 model: RoMaV2 Matcher + optional recon decoder, and loading.

Checkpoint layout is part of the interface: state dicts are keyed
``matcher.*`` / ``decoder.*``, so every checkpoint written before this
refactor still loads (and resumes) unchanged.
"""

import sys

import torch

from src.decoders import RAEDecoder
from src.paths import RESULTS_DIR, ROMAV2_CKPT, ROMAV2_SRC

if str(ROMAV2_SRC) not in sys.path:
    sys.path.insert(0, str(ROMAV2_SRC))

# RoMaV2 descriptor layout: two Descriptor layers of 1024 concatenated
DESC_LAYER_CH = 1024
DESC_CH = 2 * DESC_LAYER_CH


class DescProj(torch.nn.Module):
    """Frozen-desc baseline: linear glue in place of the trainable mv_vit.

    Same call/return interface as the mv_vit (stacked two-view input,
    x_norm_patchtokens output), so the Matcher and the mv hook are unchanged.
    The matching signal can only shape this projection + the DPT head — the
    deep encoder's contribution is exactly the gap to the mv_vit arms.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):  # (B, 2, H, W, in_dim)
        return {"x_norm_patchtokens": self.proj(x)}


class R2Model(torch.nn.Module):
    """From-scratch Matcher + (optional) recon decoder on the mv tokens."""

    def __init__(self, with_decoder, desc_only=False, skip_head=False,
                 decoder_size="B"):
        super().__init__()
        from romav2.matcher import Matcher

        self.matcher = Matcher(Matcher.Cfg(enable_amp=False))
        self.matcher.head.enable_amp = False  # vendored dpt.py patch (R2)
        if desc_only:
            self.matcher.mv_vit = DescProj(DESC_CH, self.matcher.cfg.dim)
        # recon-only desc baseline: the attention/DPT parts of the matcher
        # carry no loss, so skip their forward entirely (proj -> decoder
        # only). Params stay registered -> ckpts remain resume-compatible.
        assert not (skip_head and not desc_only)
        self.skip_head = skip_head
        self._mv = []
        self.matcher.mv_vit.register_forward_hook(
            lambda m, inp, out: self._mv.append(out["x_norm_patchtokens"])
        )
        self.decoder = (
            RAEDecoder(1024, 16, size=decoder_size) if with_decoder else None
        )

    def forward(self, desc_A, desc_B, img_A, img_B):
        if self.skip_head:
            mv_A = self.matcher.mv_vit.proj(desc_A.permute(0, 2, 3, 1))
            preds = {"mv_A": mv_A}
            if self.decoder is not None:
                preds["recon_A"] = self.decoder(
                    mv_A.permute(0, 3, 1, 2).contiguous()
                )
                mv_B = self.matcher.mv_vit.proj(desc_B.permute(0, 2, 3, 1))
                preds["recon_B"] = self.decoder(
                    mv_B.permute(0, 3, 1, 2).contiguous()
                )
            return preds

        # cached desc (B,2048,g,g) -> the Matcher's two-layer f_list format
        def to_list(d):
            d = d.permute(0, 2, 3, 1)
            return [d[..., :DESC_LAYER_CH], d[..., DESC_LAYER_CH:]]

        self._mv.clear()
        preds = self.matcher(
            to_list(desc_A),
            to_list(desc_B),
            img_A=img_A,
            img_B=img_B,
            bidirectional=True,
        )
        B, g = desc_A.shape[0], desc_A.shape[-1]
        mv = self._mv.pop().reshape(B, 2, g, g, -1)
        preds["mv_A"] = mv[:, 0]
        if self.decoder is not None:
            preds["recon_A"] = self.decoder(mv[:, 0].permute(0, 3, 1, 2).contiguous())
            preds["recon_B"] = self.decoder(mv[:, 1].permute(0, 3, 1, 2).contiguous())
        return preds


# ------------------------------------------------------------ loading


def matcher_state_dict(state, prefix="matcher."):
    """Strip the R2Model prefix off a checkpoint's matcher weights."""
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def load_pretrained_matcher(model, ckpt_path=None, verbose=True):
    """Initialize `model.matcher` from the released romav2.pt (R3 init)."""
    ckpt_path = ckpt_path or ROMAV2_CKPT
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    msd = matcher_state_dict(sd)
    model.matcher.load_state_dict(msd, strict=True)
    if verbose:
        print(
            f"matcher init from pretrained {ckpt_path.name} "
            f"({len(msd)} tensors); decoder stays random-init",
            flush=True,
        )
    return model


def run_ckpt_path(run, arm, results_dir=None):
    return (results_dir or RESULTS_DIR) / run / arm / "ckpt.pt"


def load_run_ckpt(run, arm, step=None, results_dir=None):
    """Load results/<run>/<arm>/ckpt.pt, asserting the expected final step."""
    ckpt = torch.load(
        run_ckpt_path(run, arm, results_dir), map_location="cpu", weights_only=False
    )
    if step is not None:
        assert ckpt["step"] == step, \
            f"{run}/{arm} ckpt at step {ckpt['step']}, expected {step}"
    return ckpt


def load_trained_model(run, arm, step=None, with_decoder=False, desc_only=False,
                       skip_head=False, device="cuda", results_dir=None):
    """A trained R2/R3 arm, in eval mode with grads off.

    `arm="pretrained"` loads the released romav2.pt matcher instead (the
    zero-shot baseline); with_decoder is then meaningless and must be False.
    """
    model = R2Model(with_decoder, desc_only=desc_only, skip_head=skip_head)
    if arm == "pretrained":
        assert not with_decoder, "the pretrained baseline has no recon decoder"
        load_pretrained_matcher(model, verbose=False)
    else:
        ckpt = load_run_ckpt(run, arm, step, results_dir)
        sd = ckpt["model"] if with_decoder else {
            k: v for k, v in ckpt["model"].items() if k.startswith("matcher.")
        }
        model.load_state_dict(sd, strict=True)
    return model.to(device).eval().requires_grad_(False)
