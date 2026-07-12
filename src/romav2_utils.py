"""Frozen RoMaV2 feature extraction + feature cache for the R-line (R1+).

RoMaV2 is vendored in third_party/RoMaV2. Its features are pair-conditioned:
the matcher's mv_vit jointly attends over both images, and the DPT head input
mixes image-A features with match embeddings, so every extracted feature of A
depends on the paired image B. Representations exposed per pair (A, B):

  - "dpt":  fused DPT-head map just before scratch.output_conv2,
            [B, 128, H/4, W/4]  (the head's densest feature)
  - "mv":   mv_vit output tokens for view A, [B, 1024, H/16, W/16]
            (bottleneck-like, closest to an RAE-style latent)
  - "desc": frozen DINOv3 descriptor of A (layers 11+17 concatenated),
            [B, 2048, H/16, W/16] — the matcher's own input; control that
            separates "matcher discarded appearance" from decoder capacity
  - "mvdesc": [desc ‖ mv] channel concat, [B, 3072, H/16, W/16], assembled
            at load time from the cached desc + mv (no extra precompute);
            probes the mv features' *marginal* information beyond desc

amp is disabled everywhere we can control it (Pascal GPUs); the DPT head's
internal bf16 autocast is unconditional upstream but runs fine emulated.
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROMAV2_SRC = Path(__file__).resolve().parent.parent / "third_party/RoMaV2/src"


def load_romav2_frozen(setting="turbo"):
    if str(ROMAV2_SRC) not in sys.path:
        sys.path.insert(0, str(ROMAV2_SRC))
    torch.set_float32_matmul_precision("highest")
    from romav2 import RoMaV2
    from romav2.features import Descriptor
    from romav2.matcher import Matcher

    cfg = RoMaV2.Cfg(
        descriptor=Descriptor.Cfg(enable_amp=False),
        matcher=Matcher.Cfg(enable_amp=False),
        setting=setting,
    )
    model = RoMaV2(cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def img_to_tensor01(pil_img, size):
    """PIL image -> [1,3,size,size] in [0,1] (RoMaV2 input convention)."""
    tf = transforms.Compose(
        [transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()]
    )
    return tf(pil_img.convert("RGB")).unsqueeze(0)


def cache_file(cache_dir, key, feature):
    """One .pt per pair holds dpt+mv; desc lives in a sidecar (added later)."""
    suffix = ".desc" if feature == "desc" else ""
    return Path(cache_dir) / f"{key}{suffix}.pt"


class RomaFeatureExtractor:
    """Hook-based extractor running descriptor + matcher only (no refiners)."""

    def __init__(self, model):
        self.model = model
        self._raw = {}
        model.matcher.head.scratch.output_conv2.register_forward_hook(
            lambda m, inp, out: self._raw.__setitem__("dpt", inp[0].detach())
        )
        model.matcher.mv_vit.register_forward_hook(
            lambda m, inp, out: self._raw.__setitem__(
                "mv", out["x_norm_patchtokens"].detach()
            )
        )

    @torch.no_grad()
    def extract(self, img_A, img_B):
        """img_*: [B,3,H,W] in [0,1], H/W mult. of 16 -> {"dpt","mv","desc"}."""
        B, _, H, W = img_A.shape
        f_A = self.model.f(img_A)
        f_B = self.model.f(img_B)
        self.model.matcher(f_A, f_B, img_A=img_A, img_B=img_B, bidirectional=False)
        mv = self._raw.pop("mv").reshape(B, 2, H // 16, W // 16, -1)[:, 0]
        return {
            "dpt": self._raw.pop("dpt").float(),
            "mv": mv.permute(0, 3, 1, 2).contiguous().float(),
            "desc": torch.cat(f_A, dim=-1).permute(0, 3, 1, 2).contiguous().float(),
        }


def precompute_pair_cache(
    extractor, pairs, cache_dir, size, device="cuda", features=("dpt", "mv")
):
    """Cache the given features (fp16 cpu) + anchor path per pair; skips
    existing files, so it is safe to re-run after interruption.

    pairs: list of dicts with "key", "anchor" (recon target / image A) and
    "other" (conditioning image B). All requested features go into one file
    (desc requests write the .desc sidecar, see cache_file).
    """
    #! used for the roma probe?!
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # iter over pairs
    for n, p in enumerate(pairs):
        out = cache_file(cache_dir, p["key"], features[0])
        if out.exists():
            continue
        img_A = img_to_tensor01(Image.open(p["anchor"]), size).to(device)
        img_B = img_to_tensor01(Image.open(p["other"]), size).to(device)
        feats = extractor.extract(img_A, img_B)

        payload = {k: feats[k][0].half().cpu() for k in features}
        payload["anchor"] = str(p["anchor"])
        torch.save(payload, out)
       
        if n % 50 == 0:
            print(f"  cached {n}/{len(pairs)}", flush=True)


class PairFeatureDataset(torch.utils.data.Dataset):
    """Reads cached features + reloads the anchor image as recon target.

    feature "mvdesc" concatenates [desc, mv] along channels (desc first,
    0:2048; mv 2048:3072) from the main + .desc sidecar cache files.
    """

    def __init__(self, pairs, cache_dir, feature, size):
        self.files = [cache_file(cache_dir, p["key"], feature) for p in pairs]
        self.desc_files = (
            [cache_file(cache_dir, p["key"], "desc") for p in pairs]
            if feature == "mvdesc"
            else None
        )
        self.feature = feature
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu")
        target = img_to_tensor01(Image.open(d["anchor"]), self.size)[0]
        if self.feature == "mvdesc":
            desc = torch.load(self.desc_files[i], map_location="cpu")["desc"]
            return torch.cat([desc, d["mv"]], dim=0).float(), target
        return d[self.feature].float(), target
