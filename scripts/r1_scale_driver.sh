#!/usr/bin/env bash
# R1 decoder-scale sweep (docs/R1.md appendix): does decoder capacity change
# the absolute numbers and/or the desc<->b3 gap? Runs the RAE probe at size S
# (21.6M body) and L (303M body) on the desc (ceiling control) and b3 (R2
# candidate) arms, 3cat split, same recipe as the B runs. S fits one GPU ->
# both S runs in parallel; L does not fit 11 GB -> pipelined across both
# GPUs (--pipeline), so the two L runs go sequentially. Features are already
# cached; resumable (ckpt.pt) and skips arms with metrics.json.
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
LOGDIR=results/r1_recon_probe
mkdir -p "$LOGDIR"

while pgrep -f "experiments/r1_recon_probe.py" > /dev/null; do sleep 120; done

# phase 1: S runs, one arm per GPU
gpu=0
for arm in desc b3; do
    if [ ! -f "$LOGDIR/${arm}_S/metrics.json" ]; then
        CUDA_VISIBLE_DEVICES=$gpu $PY experiments/r1_recon_probe.py \
            --feature $arm --decoder-size S \
            >> "$LOGDIR/train_${arm}_S.log" 2>&1 &
    fi
    gpu=$((gpu + 1))
done
wait

# phase 2: L runs, pipelined across both GPUs, sequential
for arm in desc b3; do
    if [ ! -f "$LOGDIR/${arm}_L/metrics.json" ]; then
        CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            $PY experiments/r1_recon_probe.py \
            --feature $arm --decoder-size L --pipeline \
            >> "$LOGDIR/train_${arm}_L.log" 2>&1
    fi
done
echo "r1 scale sweep done"
