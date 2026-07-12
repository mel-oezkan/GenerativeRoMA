"""Conv decoders reconstructing RGB from frozen feature maps (R-line)."""

import math

import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ConvDecoder(nn.Module):
    """log2(in_stride) x2-upsample stages; width halves per stage (min 64).

    Predicts RGB in [0,1] without an output activation (clamp at eval time).
    """

    def __init__(self, in_ch, in_stride, base=256):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base, 3, padding=1)]
        ch = base
        for _ in range(int(math.log2(in_stride))):
            nxt = max(ch // 2, 64)
            layers += [
                ResBlock(ch),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, nxt, 3, padding=1),
            ]
            ch = nxt
        layers += [
            ResBlock(ch),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, f):
        return self.net(f)


class ProjConvDecoder(nn.Module):
    """1x1 projection to proj_ch, then a standard ConvDecoder body.

    For concatenated multi-source features: the sources only mix inside the
    projection, so zeroing one source's channel block at eval stays
    interpretable, and the decoder body is identical to the single-source
    arms (same capacity downstream of the projection).
    """

    def __init__(self, in_ch, in_stride, proj_ch=1024, base=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, proj_ch, 1)
        self.body = ConvDecoder(proj_ch, in_stride, base)

    def forward(self, f):
        return self.body(self.proj(f))
