#!/usr/bin/env bash
# R1 self-pair control: mv features with the matcher conditioned on the
# anchor itself (A == B). Same 3cat split/recipe as scripts/r1_driver.sh;
# eval anchors coincide with the normal mv arm, so the result is directly
# comparable to results/r1_recon_probe/mv (12.09 dB). Tests whether
# cross-view conditioning destroys appearance or the matcher stages do it
# even on identical inputs.
# Waits for the hydrant-full driver to release both GPUs, shards the
# self-pair precompute across them, then trains the single arm on GPU 0.
# Resumable like the other drivers.
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
LOGDIR=results/r1_recon_probe
mkdir -p "$LOGDIR"

# wait for the hydrant driver and any probe process to release the GPUs
while pgrep -f "r1_hydrant_driver.sh" > /dev/null \
      || pgrep -f "experiments/r1_recon_probe.py" > /dev/null; do
    sleep 120
done

for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i $PY experiments/r1_recon_probe.py --precompute-only \
        --self-pair --shard "$i/2" \
        >> "$LOGDIR/precompute_selfpair_$i.log" 2>&1 &
done
wait

if [ -f "$LOGDIR/mv_selfpair/metrics.json" ]; then
    echo "mv_selfpair already done, skipping"
else
    CUDA_VISIBLE_DEVICES=0 $PY experiments/r1_recon_probe.py \
        --feature mv --self-pair \
        >> "$LOGDIR/train_mv_selfpair.log" 2>&1
fi
echo "r1 mv self-pair done"
