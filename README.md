# GenerativeRoMA

Reducing the **information/reconstruction trade-off** in dense-matching
transformer features: matching-trained encoders (RoMaV2) discard the
appearance information that generative decoding needs. We measure that
collapse and test whether a reconstruction objective — added post-hoc,
from scratch, or during fine-tuning — can prevent or undo it without
hurting matching.

See `CLAUDE.md` for environment/dataset setup and `docs/` for the full
experiment logs (`R1.md`, `R2.md`, `R3.md`, `co3d_depth_issues.md`).

## Results so far

### R1 — the collapse is real (post-hoc probes, frozen pretrained RoMaV2)

RAE-style decoders trained on frozen features of the pretrained matcher,
CO3D hydrant-full, 320 px (PSNR on held-out views):

| feature | PSNR (dB) | reading |
|---|---|---|
| desc (DINOv3 input) | **18.95** | input ceiling: appearance is there |
| mv (matcher tokens) | **13.29** | the matcher destroyed ~5.7 dB of it |
| dpt / b3 taps | between | collapse develops through the mv blocks |

### R2 v0 — from-scratch pilot (hydrant, joint matching + recon)

Stock Matcher trained from scratch, frozen precomputed desc, 6k steps
(`results/r2_v0`, docs/R2.md):

| arm | EPE | PCK5 | post-hoc mv probe |
|---|---|---|---|
| match | 5.9 | 0.66 | 16.61 dB (partly eroded already) |
| joint (recon from step 0) | 6.0 | 0.66 | **18.55 dB ≈ desc ceiling** |

Joint training keeps the tokens at the input's recon ceiling at **zero
matching cost** — at pilot scale.

### R3 — fine-tuning the *pretrained* matcher (4 full categories)

Matcher initialized from `romav2.pt` (matcher LR 2e-5, fresh RAEDecoder-B),
r3-4cat split: full hydrant + bench + toybus + toytruck, 14,236 filtered
pairs (test-frame → pose-quality ≥ 0.5 → covis floor ≥ 8/400 filter
pipeline applied before feature caching), 6k steps
(`results/r3_ft`, docs/R3.md, finished 2026-07-15):

| arm | EPE all/near/far | PCK5 all/near/far |
|---|---|---|
| pretrained (zero-shot) | 4.46 / 2.15 / 7.17 | 0.802 / 0.976 / 0.598 |
| match-ft | 2.64 / 0.82 / 4.77 | 0.884 / 0.999 / 0.750 |
| joint-ft (+recon) | 2.74 / 0.84 / 4.98 | 0.876 / 0.998 / 0.732 |

Fine-tuning roughly halves the zero-shot EPE, and adding reconstruction
costs ~nothing in matching (Δ EPE 0.1, Δ far-PCK5 1.8 pt). But the
post-hoc probes show **restoration is only partial**: match-ft mv probes
at 13.05 dB (collapse fully intact, ≈ pretrained 13.29), joint-ft at
15.40 dB — +2.35 dB restored, yet only ~40 % of the gap to the 18.95 dB
desc ceiling, vs ~93 % for from-scratch joint training (R2 v0).
**Preventing the collapse is nearly free; undoing it is hard.**

### R2 v1 — from-scratch at scale (finished 2026-07-16)

From-scratch match/joint on the r3-4cat data, 12k steps, effective batch
16 (grad accumulation 4×4); the joint arm adds a train-time GAN with
*delayed* starts (disc updates from 50%, adversarial loss from 62.5% of
training) — RAE recipe, `src/rae_gan.py` (`results/r2_v1`, docs/R2.md
"v1" section):

| arm | EPE all/near/far | PCK5 all/near/far | post-hoc mv probe |
|---|---|---|---|
| match | 7.66 / 2.02 / 14.30 | 0.678 / 0.959 / 0.347 | 15.99 dB (eroding: v0 16.61 → pretrained 13.29) |
| joint (+recon+GAN) | **7.02 / 2.02 / 12.92** | **0.686 / 0.961 / 0.361** | **19.06 dB — at the 18.95 desc ceiling** |

At scale the reconstruction objective is mildly *beneficial* for
matching (gain concentrated in far pairs), and the joint−match probe
gap widened from 1.9 dB (v0) to 3.1 dB — the collapse deepens with
matching scale exactly where the recon objective is absent.

**Frozen-DINOv3 controls at the same 12k budget** (`results/r2_v1_desc`,
mv_vit → linear 2048→1024): near-pair matching needs no deep encoder
(EPE 2.04 ≈ 2.02) — the mv_vit's contribution is entirely wide-baseline
(far EPE 20 → 13, coarse EPE 31 → 11). An equal-budget pure RAE on the
frozen features reconstructs *better* than the deep joint system
(23.7 vs 21.4 dB held-out train-time decoders; probe axis shows content
parity) and a matching-trained linear loses only 0.7 dB — **the
appearance collapse requires encoder depth; appearance itself is cheap
at every depth.**

## Running things

Every entry point is a thin Hydra script: the recipe lives in `configs/`,
the code lives in `src/`, and `experiments/*.py` only wires the two
together. A run is fully identified by its config plus an arm:

```bash
python experiments/r2_train.py experiment=r2_v1 arm=joint   # a training run
python experiments/r1_recon_probe.py feature=mv split=hydrant-full
python experiments/r3_eval_cats.py run=r3_ft arm=joint step=6000
bash scripts/r2v1_driver.sh                                 # the whole pipeline
```

Any config value can be overridden on the command line
(`optim.steps=30`, `gan.arms=[]`, `out_dir=results/tmp`). The fully
resolved config is echoed into the log and written to
`results/<run>/<arm>/run.json`, so every results directory records exactly
what produced it.

## Layout

- `configs/` — the recipes. `paths.yaml` (every filesystem root),
  `splits.yaml` (sequence splits + pair sampling), `r2_train.yaml` +
  `experiment/<run>.yaml` (one file per training run: r2_v0, r2_v1,
  r2_v2, r2_v1_desc, r3_ft), `r1_probe.yaml`, the eval/precompute
  configs, and `figures/` for the report figures.
- `src/` — the implementation. `src/r2/` (dataset + filters, model,
  losses, metrics, training loop, precompute, benchmarks), `src/r1/`
  (the RAE probe), `src/viz/` (one palette, log parsing, results IO),
  `src/paths.py`, `src/splits.py`, `src/config.py`.
- `experiments/` — entry points, one per pipeline stage:
  `r2_train.py` (matcher training/fine-tuning, all arms and splits),
  `r2_pair_covis.py` / `r2_precompute_desc.py` (filter census + per-frame
  desc cache; filters run *before* caching), `r1_recon_probe.py`
  (post-hoc recon probes — the measurement protocol),
  `r2_probe_precompute.py` (mv tokens from any trained ckpt),
  `r3_eval_cats.py` (per-category matching eval + the pretrained
  zero-shot baseline), `r2_recon_eval.py` (held-out PSNR/SSIM/LPIPS of
  the train-time decoders), `r3_bench_pose.py` (MegaDepth/ScanNet pose
  AUC), and `visualizations/` (`plot_losses.py`, `r3_report_figs.py`, …).
- `scripts/` — drivers: orchestration only (which arm, which GPU, in what
  order, what to wait for), sharing `scripts/lib.sh`. Hyperparameters
  live in `configs/`, never in a driver. Everything is resumable and
  re-runnable: finished stages are skipped, unfinished ones resume from
  their checkpoints.
