#!/usr/bin/env bash
# R2 v0 pipeline (docs/R2.md): waits for the running R1 scale sweep to
# release both GPUs, then
#   1. per-frame desc precompute, one shard per GPU (~13k frames)
#   2. 30-step GPU smoke run of the joint arm into results/r2_smoke
#      (end-to-end check of cache + dataloader + model before the long runs;
#      aborts the pipeline on failure)
#   3. real runs: match (GPU0) + joint (GPU1) in parallel, then recon (GPU0)
# Resumable: precompute skips cached frames, train resumes from ckpt.pt,
# arms with metrics.json are skipped entirely.
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
OUT=results/r2_v0
mkdir -p "$OUT"

echo "[driver $(date +%H:%M)] waiting for R1 scale sweep to finish"
while pgrep -f "r1_scale_driver.sh" > /dev/null \
   || pgrep -f "experiments/r1_recon_probe.py" > /dev/null; do
    sleep 300
done
echo "[driver $(date +%H:%M)] GPUs free, starting desc precompute"

CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_precompute_desc.py --shard 0/2 \
    >> "$OUT/precompute0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY experiments/r2_precompute_desc.py --shard 1/2 \
    >> "$OUT/precompute1.log" 2>&1 &
wait
echo "[driver $(date +%H:%M)] precompute done, smoke run"

CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_train.py --arm joint --steps 30 \
    --out-dir results/r2_smoke >> "$OUT/smoke.log" 2>&1 \
    || { echo "[driver] SMOKE RUN FAILED, aborting (see $OUT/smoke.log)"; exit 1; }
echo "[driver $(date +%H:%M)] smoke OK, launching match + joint"

if [ ! -f "$OUT/match/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_train.py --arm match \
        >> "$OUT/train_match.log" 2>&1 &
fi
if [ ! -f "$OUT/joint/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=1 $PY experiments/r2_train.py --arm joint \
        >> "$OUT/train_joint.log" 2>&1 &
fi
wait
echo "[driver $(date +%H:%M)] match+joint done, launching recon control"

if [ ! -f "$OUT/recon/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_train.py --arm recon \
        >> "$OUT/train_recon.log" 2>&1
fi
echo "[driver $(date +%H:%M)] r2 v0 pipeline done"
