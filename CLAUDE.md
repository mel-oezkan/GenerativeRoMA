# GenerativeRoMA

## Project Goal
Reduce the **information/reconstruction trade-off** in dense-matching transformer features by adding a generative (reconstruction) objective.

Two phases, in order:
1. **DINOv2 baseline** — take a pretrained DINOv2 encoder, attach a decoder, and fine-tune with a reconstruction loss. This establishes how much reconstruction quality pretrained matching/semantic features support.
2. **RoMA V2 from scratch** — train the RoMA V2 model with the reconstruction objective included *from the start* (jointly with the matching objective), so the encoder never discards the appearance information needed for reconstruction in the first place.

Sister project: `/visinf/home/lab_mozkan/computer-vision-proj-lab` (3D-aware VAE research). Shares the conda env, datasets, and the RoMA V2 source tree — see below.

## Python Environment
- `conda activate cv` (Python 3.10, env at `/visinf/home/lab_mozkan/miniconda3/envs/cv`)
- **This env is shared with `computer-vision-proj-lab`** — don't up/downgrade packages without checking that project.
- Key packages: `torch 2.5.1+cu121`, `torchvision 0.20.1`, `xformers 0.0.29`, `pytorch-lightning 2.6.0`, `hydra-core 1.3.2`, `timm 1.0.24`, `einops`, `wandb`, `pytorch-fid`, `torchmetrics`.
- A `.pth` file in the env's `site-packages` adds these to `sys.path` globally:
  - `/visinf/home/lab_mozkan/computer-vision-proj-lab/third_party` (→ `ldm`, `taming`, `co3d` tools)
  - `/visinf/home/lab_mozkan/computer-vision-proj-lab/third_party/RoMA2/src` (→ `import romav2`, v2.0.0)
  - `/visinf/home/lab_mozkan/computer-vision-proj-lab/scripts`
  - `/visinf/home/lab_mozkan/depth-anything-3/src`
- So `import romav2` works out of the box; edits to RoMA V2 code go to `computer-vision-proj-lab/third_party/RoMA2/src/romav2` (until this repo gets its own fork for the from-scratch training).
- Running scripts: the Bash tool does not preserve cwd, so always use explicit paths:
  ```
  source /visinf/home/lab_mozkan/miniconda3/etc/profile.d/conda.sh && conda activate cv && \
  PYTHONPATH=/visinf/home/lab_mozkan/GenerativeRoMA python /visinf/home/lab_mozkan/GenerativeRoMA/<script>.py
  ```

## Pretrained Weights (already on disk)
Cached in `~/.cache/torch/hub/checkpoints/`:
- `dinov2_vitb14_pretrain.pth` — DINOv2 ViT-B/14 (also `dinov2_base.ckpt`); hub repos `facebookresearch_dinov2_main` and a `dinov3` snapshot are cached too, so `torch.hub.load` works offline.
- `romav2.pt` — pretrained RoMA V2 checkpoint (reference / phase-2 comparison).
- `vgg16-397923af.pth`, `alexnet-owt-7be5be79.pth` — LPIPS backbones (LPIPS weights also via `taming` on the shared path).

## Datasets (`/visinf/projects_students/dlcv2025_groupZ`)

| Path | Size | Contents |
|------|------|----------|
| `co3d/` | 39 GB | Official CO3D format, 4 categories (`backpack`, `bench`, `car`, `toyplane`); per-sequence `images/`, `masks/`, `depths/`, `depth_masks/`, `pointcloud.ply`, plus `eval_batches/` |
| `co3d_data/` | 17 GB | ~70 CO3D categories as zips + extracted dirs (broad category coverage, few sequences each) |
| `co3d_full/` | 990 GB | Larger CO3D download (zips, partially extracted) |
| `co3d_annotations/` | small | `hydrant_{train,test,bbox}.jgz` annotation files |
| `imagenet-256/` | 8.2 GB | ImageNet at 256px, 1000 classes — single-image reconstruction training/eval |
| `imagenet-256-10k/` | 40 MB | 10k-image subset, quick evals (FID etc.) |
| `mvimgnet2/` | 42 GB | MVImgNet2 shard `mvi2_00/<class_id>/<sequence_hash>/` multi-view sequences |
| `precomputed_co3d/` | ~0 | Mostly empty (`toytruck` stub) |

**Gotcha:** `co3d_data` contains all-black placeholder frames that silently zero out matching/reconstruction losses — filter them (see `is_blank_frame` guard in the sister project).

## Suggested Workflow
1. Phase 1: DINOv2 encoder (frozen, then fine-tuned) + lightweight decoder, reconstruction loss (L1/L2 + LPIPS), train on `imagenet-256`, quick-eval FID on `imagenet-256-10k`.
2. Phase 2: RoMA V2 trained from scratch with matching + reconstruction objectives on CO3D/MVImgNet multi-view data; compare against the pretrained `romav2.pt`.
3. Track experiments with WandB; use Hydra configs (`config/*.yaml`, one per experiment) as in the sister project.
