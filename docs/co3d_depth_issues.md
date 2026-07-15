# CO3D depth-map issues (R2 warp supervision)

Documented 2026-07-14 while building R2's dense-warp GT pipeline
(`src/co3d_geom.py`, validated by `experiments/r2_warp_check.py`).
Figures: `experiments/r2_depth_report_figs.py` →
`results/r2_depth_report/`; numbers below from its `stats.json`
(120-frame random sample of hydrant train frames + the 3-sequence
near/far validation pairs).

CO3D depth maps are **COLMAP MVS reconstructions**, not sensor depth.
Three concrete problems follow from that, plus one lookalike that is not
a problem. All were found on the hydrant category; the mechanisms are
category-independent.

## 1. `test_*` frames ship all-zero depth (fig1)

CO3D withholds evaluation-set data: every frame whose
`meta.frame_type` contains `test` has a depth PNG of all zeros, and
`test_unseen` frames additionally ship an **all-black RGB image** (the
challenge placeholder — the same family as the blank-frame gotcha known
from `co3d_data`). Naively using such frames yields zero covisibility
and silently supervises nothing.

- Hydrant census: 504 of 135,878 frames (0.37%) are `test_*`
  (352 `test_known` + 152 `test_unseen`); `dev_*` frames have depth.
- Impact on R2: pairs touching test frames are dropped from matching
  supervision (10 of 6631 train pairs; all 60 eval pairs unaffected).
- Guard: `frame_type` check in `r2_train.has_depth()` + a
  `covis.sum()` floor at loss time.

## 2. MVS noise: holes, and reliability masks that keep only the object (fig2)

Two layers here. The raw depth maps (`depth > 0`) cover 85–99% of
pixels — but they are noisy MVS estimates, worst on far-field ground,
texture-poor/specular surfaces, and borders. CO3D's own `depth_masks`
(reliability masks) are far stricter: after masking, **usable depth
covers a median of only ~12% of pixels** (120-frame train sample: p10
3.2%, median 12.0%, max 31.3%) — essentially the object plus a little
nearby ground, speckled with holes even there. And the worst sampled
*train* frame has 0.0% valid depth (fully empty depth PNG despite
being `train_known`), so the `frame_type` guard from problem 1 is not
sufficient on its own.

What survives the mask is accurate, though: near-pair photometric
validation reads 24–31 dB masked PSNR.

- Impact: supervision is object-centric and sparse — covisibility is
  ~12–13% of pixels for near pairs, ~2–3% for far pairs at the tight
  threshold. Fine for masked losses; do not expect dense background
  supervision. Matching on background must come from the model
  generalizing, not from GT.
- Guard: `depth > 0 ∧ depth_mask` everywhere; losses average only over
  covisible pixels with a `clamp(min=1)` denominator, which also
  absorbs the fully-empty train frames.

## 3. See-through covisibility at loose depth thresholds (fig3, fig4)

The covisibility check compares the depth of an unprojected A-pixel,
reprojected into B, against B's own depth map — relative tolerance
`rel_thresh`. **CO3D's scene scale breaks the usual defaults**: cameras
sit ~15 depth units from ~1-unit-sized objects, so a "loose" 0.10
relative tolerance is ±1.5 units — thicker than the object. Back-side
surfaces then pass the check *through* the object and get marked
covisible with front-side pixels.

- At `rel_thresh 0.10` on the validation far pair, **59.7% of the
  covisible set is false** (back-side): covis 6.2% of pixels at 0.10
  vs 2.5% at 0.01, with mean photometric error 0.224 in the false
  region vs 0.167 in the kept region (fig3, bottom-right panel).
- Near pairs are insensitive (0.124→0.127 covis across 0.005→0.10):
  both views see the same surface, so the check only fails through
  disagreement, not occlusion.
- Fix: `rel_thresh = 0.01` (±0.15 units), safely above MVS noise but
  below object thickness. This is the default in
  `co3d_geom.compute_warp`.
- General lesson: relative depth thresholds implicitly assume
  scene-scale ≈ object-scale; for object-centric captures with distant
  cameras they must be far tighter than the 5–10% used on scene
  datasets (ScanNet/MegaDepth-style).

## Not a problem: far-pair photometric residual (fig5)

Far pairs read only ~14 dB masked photometric PSNR even with correct
geometry — the blend in fig5 is structurally aligned but differs in
shading/exposure (non-Lambertian paint, auto-exposure/white-balance
drift, sun angle). This is a property of wide-baseline real captures,
not depth error. Consequence: photometric error is **not** a usable
supervision or QA signal on far pairs; geometric (warp-space) losses
and metrics are.

## Filtering pipeline adopted (2026-07-14)

`r2_train.build_train_pairs` (shared with the desc precompute; all arms
train on the identical pair set; eval pairs are never filtered):

1. **test-frame filter** — drop pairs touching `test_*` frames:
   6631 → 6621 train pairs.
2. **sequence pose-quality floor** — `viewpoint_quality_score ≥ 0.5`
   (NaN → drop) from `sequence_annotations.jgz`: 6621 → 6425. This is the
   only guard against *systematically* wrong warps: the depth-consistency
   check reuses the same bad poses, so it passes them. All 6 eval
   sequences score ≥ 0.75 and are unaffected.
3. **pair covisibility floor** — ≥ 8/400 covisible tokens at /16 in the
   better direction (census: `experiments/r2_pair_covis.py` →
   `r2_meta/pair_covis_hydrant.json`): 6425 → **4201** (−355 near,
   −1869 far). Census medians: near pairs 36 covisible tokens, far pairs
   **6** — hydrant far views are near-opposite sides, most genuinely share
   almost no surface. The dropped pairs carry ~5% of token-level matching
   supervision at ~a third of the compute, and include the
   empty-depth population whose confidence target would be wrong
   ("unknown" ≠ "not covisible"). Overlap-based pair selection matches
   MegaDepth-style matcher training. Trade-off: train mix shifts from
   50/50 to ~60/40 near/far. Distribution: fig6
   (`r2_covis_census_fig.py`).

Deliberately *not* filtered: blur/exposure (unvalidated benefit, breaks
R1 comparability, matching wants hard frames) and low-covis-but-valid far
pairs above the floor (wide-baseline signal).