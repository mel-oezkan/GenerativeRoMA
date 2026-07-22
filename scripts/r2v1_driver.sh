#!/usr/bin/env bash
# R2 v1: from-scratch matcher training scaled up from the v0 pilot.
# Same r3-4cat data/filters/desc-cache as R3 (census + precompute already
# done by r3_driver.sh — this driver only trains):
#   - 4 categories (hydrant/bench/toybus/toytruck), 14236 filtered pairs
#   - 12000 steps, batch 4 x accum 4 = effective batch 16
#   - joint arm adds the train-time GAN (RAE recipe) with DELAYED starts:
#     disc updates from 50%, adversarial loss from 62.5% of training
#     (v0 had no train-time GAN at all)
# match (GPU0) + joint (GPU1) in parallel; resumable via ckpt.pt; arms
# with metrics.json are skipped.
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
OUT=results/r2_v1
SPLIT=r3-4cat
STEPS=12000
ACCUM=4
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[driver $(date +%H:%M)] launching r2 v1 match + joint (from scratch)"
if [ ! -f "$OUT/match/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_train.py --arm match \
        --split $SPLIT --steps $STEPS --accum $ACCUM \
        --out-dir "$OUT" >> "$OUT/train_match.log" 2>&1 &
fi
if [ ! -f "$OUT/joint/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=1 $PY experiments/r2_train.py --arm joint \
        --split $SPLIT --steps $STEPS --accum $ACCUM --gan \
        --out-dir "$OUT" >> "$OUT/train_joint.log" 2>&1 &
fi
wait
echo "[driver $(date +%H:%M)] r2 v1 done"
