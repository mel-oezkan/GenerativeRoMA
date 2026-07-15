"""RAE stage-1 GAN components, vendored from github.com/bytetriper/RAE
(src/disc/{dinodisc,diffaug,gan_loss}.py, src/train_stage1.py) for the R1
probe v3 decoder training (arXiv:2510.11690, appendix C.2 / Table 12).

Discriminator: StyleGAN-T design with a *frozen* DINO ViT-S/8 feature
network (paper: S/8 instead of S/16 stabilizes training and avoids
adversarial patches), standard batch norm instead of virtual BN, spectral-
norm conv1d heads (kernel 9) on the token sequences of blocks 2/5/8/11 +
final, inputs bilinearly resized to 224x224. Hinge D loss / vanilla G loss
(VQGAN), DiffAug (StyleGAN-T, prob 1.0, cutout 0.0) before the
discriminator, and the VQGAN adaptive GAN weight on the decoder's last
layer. Simplifications vs upstream: flash-attn/fused-MLP fallbacks and the
dead RandomWindowCrop path (crop_prob<0) removed; manual attention replaced
by F.scaled_dot_product_attention (same math).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm

DINO_S8_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"


# ---------------------------------------------------------------- GAN losses

def hinge_d_loss(logits_real, logits_fake):
    """Hinge discriminator loss used by VQGAN."""
    return 0.5 * (
        F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean()
    )


def vanilla_g_loss(logits_fake):
    """Original GAN generator loss."""
    return -logits_fake.mean()


def calculate_adaptive_weight(recon_loss, gan_loss, layer, max_d_weight=1e4):
    """VQGAN (Esser et al. 2021) adaptive lambda balancing recon and GAN
    gradients at the decoder's last layer."""
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    return torch.clamp(d_weight, 0.0, max_d_weight).detach()


# ------------------------------------------------------- frozen DINO ViT-S/8

class _Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        B, L, C = x.shape
        q, k, v = (
            self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4).unbind(0)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, L, C))


class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _SABlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, norm_eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = _Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = _Mlp(embed_dim, round(embed_dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class FrozenDINO(nn.Module):
    """DINO ViT-S/8 feature network; input in [-1,1], resized to 224.
    Returns [final, block2, block5, block8, block11] token activations
    as [B, C, N]."""

    def __init__(self, depth=12, key_depths=(2, 5, 8, 11), norm_eps=1e-6,
                 patch_size=8, embed_dim=384, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = 224
        self.patch_nums = self.img_size // patch_size
        self.patch_embed = _PatchEmbed(patch_size, 3, embed_dim)
        # x in [-1,1] -> ImageNet-normalized
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer("x_scale", (0.5 / std).reshape(1, 3, 1, 1))
        self.register_buffer("x_shift", ((0.5 - mean) / std).reshape(1, 3, 1, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_nums**2 + 1, embed_dim))
        self.key_depths = set(key_depths)
        self.blocks = nn.Sequential(
            *[_SABlock(embed_dim, num_heads, mlp_ratio, norm_eps) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)  # loaded, unused (as upstream)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        x = x * self.x_scale + self.x_shift
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        activations = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.key_depths:
                activations.append(x[:, 1:, :].transpose(1, 2))
        activations.insert(0, x[:, 1:, :].transpose(1, 2))
        return activations


# ------------------------------------------------------- discriminator heads

class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.ratio = 1 / np.sqrt(2)

    def forward(self, x):
        return (self.fn(x).add(x)).mul_(self.ratio)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    """Upstream's replacement for StyleGAN-T's virtual batch norm
    (virtual_bs=1 in the shipped code): per-sample, per-channel statistics
    over the token dimension, no running stats."""

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = x.size()
        x = x.float().view(x.size(0), -1, x.size(-2), x.size(-1))  # [B,1,C,N]
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight[None, :, None] + self.bias[None, :, None]
        return x.view(shape)


def make_block(channels, kernel_size, norm_eps):
    return nn.Sequential(
        SpectralConv1d(channels, channels, kernel_size=kernel_size,
                       padding=kernel_size // 2, padding_mode="circular"),
        BatchNormLocal(channels, eps=norm_eps),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class DinoDisc(nn.Module):
    """Frozen DINO-S/8 + one spectral-norm conv head per tapped depth;
    returns concatenated per-token logits [B, 5*784]."""

    def __init__(self, dino_ckpt_path, ks=9, key_depths=(2, 5, 8, 11), norm_eps=1e-6):
        super().__init__()
        state = torch.load(dino_ckpt_path, map_location="cpu")
        for key in sorted(state.keys()):  # zero k-bias, as upstream
            if ".attn.qkv.bias" in key:
                bias = state[key]
                C = bias.numel() // 3
                bias[C:2 * C].zero_()
        dino = FrozenDINO(key_depths=key_depths, norm_eps=norm_eps)
        missing, unexpected = dino.load_state_dict(state, strict=False)
        missing = [m for m in missing if all(x not in m for x in ("x_scale", "x_shift"))]
        if missing or unexpected:
            raise RuntimeError(f"DINO ckpt mismatch: missing={missing} unexpected={unexpected}")
        self.dino_proxy = (dino,)  # tuple hides it from .parameters()/.to() recursion
        dino_C = dino.embed_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    make_block(dino_C, kernel_size=1, norm_eps=norm_eps),
                    ResidualBlock(make_block(dino_C, kernel_size=ks, norm_eps=norm_eps)),
                    SpectralConv1d(dino_C, 1, kernel_size=1, padding=0),
                )
                for _ in range(len(key_depths) + 1)
            ]
        )

    def to(self, *args, **kwargs):
        self.dino_proxy[0].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x_in_pm1):
        activations = self.dino_proxy[0](x_in_pm1)
        return torch.cat(
            [head(act).view(x_in_pm1.shape[0], -1)
             for head, act in zip(self.heads, activations)],
            dim=1,
        )


# -------------------------------------------------------------------- DiffAug

class DiffAug:
    """StyleGAN-T differentiable augmentation (Zhao et al. 2020): per-batch
    random translation (1/8) and color jitter; cutout disabled per the RAE
    config (prob=1.0, cutout=0.0)."""

    def __init__(self, prob=1.0, cutout=0.0):
        self.grids = {}
        self.prob = abs(prob)
        self.using_cutout = prob > 0
        self.cutout = cutout

    def get_grids(self, B, x, y, dev):
        if (B, x, y) not in self.grids:
            self.grids[(B, x, y)] = torch.meshgrid(
                torch.arange(B, dtype=torch.long, device=dev),
                torch.arange(x, dtype=torch.long, device=dev),
                torch.arange(y, dtype=torch.long, device=dev),
                indexing="ij",
            )
        return self.grids[(B, x, y)]

    def aug(self, BCHW):
        if BCHW.dtype != torch.float32:
            BCHW = BCHW.float()
        if self.prob < 1e-6:
            return BCHW
        trans, color, cut = (torch.rand(3) <= self.prob).tolist()
        B, dev = BCHW.shape[0], BCHW.device
        rand01 = torch.rand(7, B, 1, 1, device=dev) if (trans or color or cut) else None

        raw_h, raw_w = BCHW.shape[-2:]
        if trans:
            ratio = 0.125
            delta_h, delta_w = round(raw_h * ratio), round(raw_w * ratio)
            translation_h = rand01[0].mul(2 * delta_h + 1).floor().long() - delta_h
            translation_w = rand01[1].mul(2 * delta_w + 1).floor().long() - delta_w
            grid_B, grid_h, grid_w = self.get_grids(B, raw_h, raw_w, dev)
            grid_h = (grid_h + translation_h).add_(1).clamp_(0, raw_h + 1)
            grid_w = (grid_w + translation_w).add_(1).clamp_(0, raw_w + 1)
            bchw_pad = F.pad(BCHW, [1, 1, 1, 1, 0, 0, 0, 0])
            BCHW = bchw_pad.permute(0, 2, 3, 1).contiguous()[grid_B, grid_h, grid_w] \
                .permute(0, 3, 1, 2).contiguous()

        if color:
            BCHW = BCHW.add(rand01[2].unsqueeze(-1).sub(0.5))
            bchw_mean = BCHW.mean(dim=1, keepdim=True)
            BCHW = BCHW.sub(bchw_mean).mul(rand01[3].unsqueeze(-1).mul(2)).add_(bchw_mean)
            bchw_mean = BCHW.mean(dim=(1, 2, 3), keepdim=True)
            BCHW = BCHW.sub(bchw_mean).mul(rand01[4].unsqueeze(-1).add(0.5)).add_(bchw_mean)

        if self.using_cutout and self.cutout > 0 and cut:
            cutout_h, cutout_w = round(raw_h * self.cutout), round(raw_w * self.cutout)
            offset_h = rand01[5].mul(raw_h + (1 - cutout_h % 2)).floor().long()
            offset_w = rand01[6].mul(raw_w + (1 - cutout_w % 2)).floor().long()
            grid_B, grid_h, grid_w = self.get_grids(B, cutout_h, cutout_w, dev)
            grid_h = (grid_h + offset_h).sub_(cutout_h // 2).clamp(min=0, max=raw_h - 1)
            grid_w = (grid_w + offset_w).sub_(cutout_w // 2).clamp(min=0, max=raw_w - 1)
            mask = torch.ones(B, raw_h, raw_w, dtype=BCHW.dtype, device=dev)
            mask[grid_B, grid_h, grid_w] = 0
            BCHW = BCHW.mul(mask.unsqueeze(1))

        return BCHW
