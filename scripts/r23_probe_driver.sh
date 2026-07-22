#!/usr/bin/env bash
# Post-hoc measurement pass over the R3 fine-tune and R2 v1 checkpoints
# (docs/R3.md, docs/R2.md "v1"):
#   0. per-category matching eval (r3_eval_cats.py) for r3_ft match/joint,
#      the pretrained zero-shot baseline, and r2_v1 match/joint
#   1. desc fill for the hydrant-full probe pairs (no-op if r2_desc_320
#      is already complete from the R2 v0 probe pass)
#   2. mv extraction from all 4 ckpts (hydrant-full probe pairs, so PSNR
#      stays comparable to stock RoMaV2 13.29 dB / desc ceiling 18.95 dB)
#   3. R1 v3 RAE probe per arm (fresh RAEDecoder-B), 2 arms per wave
# Resumable: extraction skips cached pairs, probes resume from ckpt.pt,
# finished probes (metrics.json) are skipped.
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
FEATS=/visinf/projects_students/dlcv2025_groupZ/romav2_feats
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[driver $(date +%H:%M)] per-category evals"
mkdir -p results/r3_probe results/r2_v1_probe
if [ ! -f results/r3_ft/pretrained/metrics_per_cat.json ]; then
    CUDA_VISIBLE_DEVICES=0 $PY experiments/r3_eval_cats.py --run r3_ft \
        --arm pretrained >> results/r3_ft/eval_cats.log 2>&1
fi
for arm in match joint; do
    if [ ! -f results/r3_ft/$arm/metrics_per_cat.json ]; then
        CUDA_VISIBLE_DEVICES=0 $PY experiments/r3_eval_cats.py --run r3_ft \
            --arm $arm --step 6000 >> results/r3_ft/eval_cats.log 2>&1 &
    fi
    if [ ! -f results/r2_v1/$arm/metrics_per_cat.json ]; then
        CUDA_VISIBLE_DEVICES=1 $PY experiments/r3_eval_cats.py --run r2_v1 \
            --arm $arm --step 12000 >> results/r2_v1/eval_cats.log 2>&1 &
    fi
    wait
done

echo "[driver $(date +%H:%M)] desc fill"
CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_probe_precompute.py --fill-desc \
    >> results/r3_probe/desc_fill.log 2>&1

echo "[driver $(date +%H:%M)] mv extraction r3_ft (match GPU0, joint GPU1)"
CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_probe_precompute.py --run r3_ft \
    --arm match --step 6000 --cache-name r3_probe_320 \
    >> results/r3_probe/extract_match.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY experiments/r2_probe_precompute.py --run r3_ft \
    --arm joint --step 6000 --cache-name r3_probe_320 \
    >> results/r3_probe/extract_joint.log 2>&1 &
wait

echo "[driver $(date +%H:%M)] mv extraction r2_v1 (match GPU0, joint GPU1)"
CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_probe_precompute.py --run r2_v1 \
    --arm match --step 12000 --cache-name r2v1_probe_320 \
    >> results/r2_v1_probe/extract_match.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY experiments/r2_probe_precompute.py --run r2_v1 \
    --arm joint --step 12000 --cache-name r2v1_probe_320 \
    >> results/r2_v1_probe/extract_joint.log 2>&1 &
wait

echo "[driver $(date +%H:%M)] probe wave 1: r3_ft (match GPU0, joint GPU1)"
for arm in match joint; do
    gpu=$([ $arm = match ] && echo 0 || echo 1)
    if [ ! -f results/r3_probe/$arm/metrics.json ]; then
        CUDA_VISIBLE_DEVICES=$gpu $PY experiments/r1_recon_probe.py \
            --feature mv --split hydrant-full \
            --cache-dir "$FEATS/r3_probe_320/$arm" \
            --out-dir results/r3_probe/$arm \
            >> results/r3_probe/train_$arm.log 2>&1 &
    fi
done
wait

echo "[driver $(date +%H:%M)] probe wave 2: r2_v1 (match GPU0, joint GPU1)"
for arm in match joint; do
    gpu=$([ $arm = match ] && echo 0 || echo 1)
    if [ ! -f results/r2_v1_probe/$arm/metrics.json ]; then
        CUDA_VISIBLE_DEVICES=$gpu $PY experiments/r1_recon_probe.py \
            --feature mv --split hydrant-full \
            --cache-dir "$FEATS/r2v1_probe_320/$arm" \
            --out-dir results/r2_v1_probe/$arm \
            >> results/r2_v1_probe/train_$arm.log 2>&1 &
    fi
done
wait
echo "[driver $(date +%H:%M)] r23 probe pass done"
