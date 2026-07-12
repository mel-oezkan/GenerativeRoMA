#!/usr/bin/env bash
# R1 recon probe: precompute the shared feature cache (GPU0), then train one
# decoder per feature arm, alternating GPUs. Arms with a metrics.json are
# skipped, so the script is safe to re-run after adding arms (mvdesc added
# 2026-07-12; it reuses the cached desc+mv, no new precompute).
# Resumable: re-run after interruption (cache skips done pairs, ckpt resumes).
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
LOGDIR=results/r1_recon_probe
mkdir -p "$LOGDIR"

CUDA_VISIBLE_DEVICES=0 $PY experiments/r1_recon_probe.py --precompute-only \
    2>&1 | tee -a "$LOGDIR/precompute.log"

gpu=0
for feat in dpt mv desc mvdesc; do
    if [ -f "$LOGDIR/$feat/metrics.json" ]; then
        echo "$feat already done, skipping"
        continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu $PY experiments/r1_recon_probe.py --feature "$feat" \
        >> "$LOGDIR/train_$feat.log" 2>&1 &
    gpu=$(( (gpu + 1) % 2 ))
done
wait
echo "r1 done"
