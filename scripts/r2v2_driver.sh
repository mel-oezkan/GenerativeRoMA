#!/usr/bin/env bash
# R2 v2: same as the r2_v1 joint run, ONLY change = reconstruct BOTH views.
# The train-time recon loss now decodes mv_A -> image A and mv_B -> image B
# and *averages* the two (L1 + LPIPS + GAN), so the recon-vs-matching weight
# is identical to v1 -- the sole difference is that view B's mv tokens now
# get recon gradient too (v1 decoded view A only). See docs/R2.md.
#
# Hparams identical to r2_v1/joint (r3-4cat, 12000 steps, effective batch 16,
# delayed GAN: disc 50% / adversarial 62.5%) EXCEPT the micro-batch: two
# decoder+LPIPS+GAN passes per step OOM the 11 GB 1080 Ti at batch 4, so
# batch 2 x accum 8 (= same effective batch 16, same optimizer trajectory).
#
# Only the joint arm is (re)trained; the match baseline is r2_v1/match.
# Resumable via ckpt.pt; skipped if metrics.json already exists.
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
OUT=results/r2_v2
SPLIT=r3-4cat
STEPS=12000
BATCH=2
ACCUM=8
GPU=${GPU:-1}
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[driver $(date +%H:%M)] launching r2 v2 joint (two-view recon) on GPU $GPU"
if [ ! -f "$OUT/joint/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY experiments/r2_train.py --arm joint \
        --split $SPLIT --steps $STEPS --batch $BATCH --accum $ACCUM --gan \
        --out-dir "$OUT" >> "$OUT/train_joint.log" 2>&1
fi
echo "[driver $(date +%H:%M)] r2 v2 done"
