"""RAE decoder reconstructing RGB from frozen feature maps (R-line, probe v3).

Faithful port of the RAE stage-1 decoder ("Diffusion Transformers with
Representation Autoencoders", arXiv:2510.11690; github.com/bytetriper/RAE,
src/stage1/decoders/decoder.py `GeneralDecoder` = the HF ViT-MAE decoder):
linear embed -> trainable [CLS] + fixed 2D sin-cos pos-embed -> pre-LN ViT
blocks (GELU MLP x4, eps 1e-12) -> LayerNorm -> linear patch head -> drop
CLS -> unpatchify. Sizes follow paper Table 13.

Deviations from RAE, forced by the probe setting (see docs/R1.md):
- Input is a feature *grid*, not encoder tokens: features finer than /16
  (dpt @ /4) are folded PATCH/in_stride blocks -> one token, because the
  paper rule p_d = p_e would mean 6400 tokens @ /4 (infeasible here).
- RAE cancels the affine of the encoder's final LayerNorm (paper C.1) so
  the decoder sees normalized tokens; RoMaV2 taps are raw feature maps, so
  the equivalent non-affine LayerNorm is applied here on the input tokens
  (per source slice for concatenated mvdesc inputs, keeping the zeroed-
  source eval ablation interpretable: LN(0) = 0).
"""

import torch
import torch.nn as nn
from einops import rearrange

# paper Table 13: width, heads, depth (MLP ratio 4)
SIZES = {
    "S": (384, 6, 12),
    "B": (768, 12, 12),
    "L": (1024, 16, 24),
    "XL": (1152, 16, 28),
}


def sincos_pos_embed_2d(width, h, w):
    """Fixed 2D sin-cos positional embedding, [h*w, width] (MAE-style)."""

    def _1d(dim, pos):
        omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim // 2))
        out = pos.flatten()[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)

    gh, gw = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    return torch.cat([_1d(width // 2, gh), _1d(width // 2, gw)], dim=1)


class Block(nn.Module):
    """Pre-LN transformer block (MSA + MLP), HF ViT-MAE layer semantics."""

    def __init__(self, width, heads, eps=1e-12):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=eps)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


def noise_augment(z, tau, generator=None):
    """RAE noise-augmented decoding (paper 4.3): z + n, n ~ N(0, sigma^2),
    sigma ~ |N(0, tau^2)| per sample. RAE stage-1 default tau = 0.8; the R1
    probe keeps tau = 0 (noise robustness only matters for diffusion
    sampling and would blur the reconstructability read-out)."""
    if tau <= 0:
        return z
    sigma = (torch.randn(z.shape[0], generator=generator, device=z.device) * tau).abs()
    return z + sigma.view(-1, *([1] * (z.dim() - 1))) * torch.randn_like(z)


class RAEDecoder(nn.Module):
    """Feature grid @ in_stride -> tokens @ /PATCH -> RAE ViT decoder -> RGB.

    For concatenated multi-source features (mvdesc) pass `ln_slices` with
    one (start, end) channel range per source: each source is layer-normed
    separately and the embed linear doubles as the mixing projection, so
    zeroing one source's channel block at eval stays interpretable.

    Predicts RGB in [0,1] without an output activation (clamp at eval time).
    """

    PATCH = 16

    def __init__(self, in_ch, in_stride, size="B", ln_slices=None, noise_tau=0.0):
        super().__init__()
        assert self.PATCH % in_stride == 0, "in_stride must divide PATCH"
        width, heads, depth = SIZES[size]
        self.group = self.PATCH // in_stride
        self.noise_tau = noise_tau
        self.ln_slices = ln_slices or [(0, in_ch)]
        self.input_norms = nn.ModuleList(
            [nn.LayerNorm(e - s, eps=1e-6, elementwise_affine=False)
             for s, e in self.ln_slices]
        )
        self.decoder_embed = nn.Linear(in_ch * self.group**2, width)
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.decoder_norm = nn.LayerNorm(width, eps=1e-12)
        self.decoder_pred = nn.Linear(width, 3 * self.PATCH**2)
        self._pos = None  # cached [1, 1+h*w, width] (zero row for CLS)

    @property
    def last_layer(self):
        """Anchor for the VQGAN-style adaptive GAN weight."""
        return self.decoder_pred.weight

    def pipeline(self, second_device, first_blocks=None):
        """Model-parallel split for decoders too big for one GPU (L/XL on
        11 GB 1080 Tis): blocks after `first_blocks` + norm + patch head
        move to `second_device`; embed/CLS/first blocks stay put. forward()
        returns on the input device, so callers (losses, disc, EMA) are
        unaffected. Default split is heavily asymmetric (1/6 of the blocks
        on the input device): that device also hosts the discriminator,
        LPIPS-VGG, and the GAN-phase autograd graph — the desc_L run OOMed
        on cuda:0 at GAN start with both a 1/2 and a 1/3 split."""
        half = first_blocks if first_blocks is not None else max(1, len(self.blocks) // 6)
        for blk in self.blocks[half:]:
            blk.to(second_device)
        self.decoder_norm.to(second_device)
        self.decoder_pred.to(second_device)
        return self

    def forward(self, f):
        in_device = f.device
        # per-source non-affine LN on raw tokens (channel dim), then fold
        f = torch.cat(
            [ln(f[:, s:e].movedim(1, -1)).movedim(-1, 1)
             for (s, e), ln in zip(self.ln_slices, self.input_norms)],
            dim=1,
        )
        if self.training and self.noise_tau > 0:  # on the normalized latent, as in RAE
            f = noise_augment(f, self.noise_tau)
        g = self.group
        x = rearrange(f, "b c (h p) (w q) -> b (h w) (p q c)", p=g, q=g)
        h, w = f.shape[-2] // g, f.shape[-1] // g
        x = self.decoder_embed(x)
        if self._pos is None or self._pos.shape[1] != 1 + h * w:
            grid = sincos_pos_embed_2d(x.shape[-1], h, w)
            self._pos = torch.cat([torch.zeros(1, x.shape[-1]), grid])[None]
        x = torch.cat([self.trainable_cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self._pos.to(x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x.to(next(blk.parameters()).device))  # no-op unless pipelined
        x = x.to(self.decoder_norm.weight.device)
        x = self.decoder_pred(self.decoder_norm(x))[:, 1:]  # drop CLS
        return rearrange(
            x, "b (h w) (p q c) -> b c (h p) (w q)", h=h, w=w, p=self.PATCH, q=self.PATCH
        ).to(in_device)
