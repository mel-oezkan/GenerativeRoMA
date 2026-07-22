#!/usr/bin/env bash
# Frozen-DINOv3 control for R2 v1 (docs/R2.md): same data (r3-4cat), same
# losses/steps/accum/GAN recipe as r2_v1, but the mv_vit is replaced by a
# linear 2048->1024 projection (--desc-baseline) — only the projection,
# the DPT head and (joint) the decoder train. The gap to the r2_v1 arms
# is the contribution of the deep trained encoder; the recon side of this
# baseline is the R1 desc ceiling (18.95 dB) by construction.
# Waits for the r23 probe pass to free the GPUs, then match (GPU0) +
# joint+GAN (GPU1) -> results/r2_v1_desc. Resumable as usual.
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
OUT=results/r2_v1_desc
SPLIT=r3-4cat
STEPS=12000
ACCUM=4
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[driver $(date +%H:%M)] waiting for the r23 probe pass to finish"
while [ ! -f results/r2_v1_probe/match/metrics.json ] || \
      [ ! -f results/r2_v1_probe/joint/metrics.json ]; do
    sleep 300
done

echo "[driver $(date +%H:%M)] GPUs free, smoke run (30 steps, desc-joint)"
if [ ! -f results/r2_v1_desc_smoke/joint/metrics.json ]; then
    CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_train.py --arm joint --steps 30 \
        --split $SPLIT --desc-baseline \
        --out-dir results/r2_v1_desc_smoke >> "$OUT/smoke.log" 2>&1 \
        || { echo "[driver] SMOKE RUN FAILED, aborting (see $OUT/smoke.log)"; exit 1; }
fi
echo "[driver $(date +%H:%M)] smoke OK, launching desc-match + desc-joint"

if [ ! -f "$OUT/match/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_train.py --arm match \
        --split $SPLIT --steps $STEPS --accum $ACCUM --desc-baseline \
        --out-dir "$OUT" >> "$OUT/train_match.log" 2>&1 &
fi
if [ ! -f "$OUT/joint/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=1 $PY experiments/r2_train.py --arm joint \
        --split $SPLIT --steps $STEPS --accum $ACCUM --gan --desc-baseline \
        --out-dir "$OUT" >> "$OUT/train_joint.log" 2>&1 &
fi
wait
echo "[driver $(date +%H:%M)] r2 v1 desc-baseline done"
