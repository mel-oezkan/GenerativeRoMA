# R3 in brief — joint fine-tuning of the pretrained RoMaV2 checkpoint

**Question.** The released `romav2.pt` matcher has collapsed its mv tokens to
matching-only content (R1: 13.29 dB probe PSNR vs 18.95 dB from the DINOv3
descriptors it consumes). Can a reconstruction objective *undo* that collapse
in an already-converged matcher, and what does it cost in matching?

**Setup.** Init from `romav2.pt` (strict, 216 tensors) + random-init
RAEDecoder-B. 4 full CO3D categories (hydrant, bench, toybus, toytruck),
14,236 train pairs, 6k steps, batch 4, matcher LR 2e-5 / decoder 2e-4.
Two arms: `ft-match` (matching losses only, the control) and `ft-joint`
(+ L1 + LPIPS recon on the mv_A tokens, λ_rec = 1.0, no GAN).
Eval: 209 held-out pairs, coarse /4 warp at 320 px.

## Matching

| arm | EPE all/near/far | PCK1 | PCK5 all/near/far |
|---|---|---|---|
| pretrained (zero-shot) | 4.46 / 2.15 / 7.17 | 0.109 | 0.802 / 0.976 / 0.598 |
| ft-match | **2.64** / 0.82 / **4.77** | **0.441** | **0.884** / 0.999 / **0.750** |
| ft-joint (+recon) | 2.74 / 0.84 / 4.98 | 0.433 | 0.876 / 0.998 / 0.732 |

- In-domain fine-tuning roughly halves zero-shot EPE and quadruples PCK1.
- **Recon is nearly free**: ft-joint gives up ~0.10 EPE and ~1.8 pt far-PCK5
  against the control. The loss curves of the two arms overlap almost exactly.
- The gap is uniform across all four categories (0.05–0.19 EPE) — no category
  absorbs the cost.

## Reconstruction (post-hoc R1 probe, fresh decoder on frozen tokens)

| mv features from | PSNR | SSIM | LPIPS |
|---|---|---|---|
| pretrained (R1) | 13.29 | 0.141 | 0.631 |
| ft-match | 13.05 | 0.141 | 0.613 |
| **ft-joint** | **15.40** | 0.196 | 0.502 |
| scratch-joint 6k (R2 v0) | 18.55 | 0.277 | 0.292 |
| desc ceiling (R1) | 18.95 | 0.292 | 0.282 |

- **Matching-only fine-tuning leaves the collapse fully intact** (13.05 ≈ 13.29):
  the collapsed representation is a stable point of the matching objective.
- **The recon objective claws back +2.35 dB** over the control — real
  restoration, but only ~40 % of the gap to the descriptor ceiling, versus
  ~93 % for from-scratch joint training at the *same* 6k budget.

## Headline

**Prevention ≫ cure.** Adding reconstruction costs essentially nothing in
matching accuracy whether you train from scratch or fine-tune, but recovering
appearance from a converged matcher only partially works: the recon gradient
has to undo a learned invariance at a matcher LR (2e-5) deliberately chosen to
protect matching. Training jointly from step 0 reaches the ceiling; retrofitting
gets you 40 % of the way.

**Open lever.** Matcher LR and step budget were not swept — whether hotter or
longer joint fine-tuning closes the rest of the gap (and at what matching cost)
is the natural follow-up before calling the asymmetry fundamental.

Full details, per-category tables, figures and caveats: [docs/R3.md](R3.md).
