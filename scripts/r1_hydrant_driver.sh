#!/usr/bin/env bash
# R1 hydrant-full: same probe/recipe as scripts/r1_driver.sh but on the
# --split hydrant-full data (all 726 extracted hydrant sequences, 10 train
# pairs/seq; eval = the same 6 hydrant sequences as the 3cat split).
# Precompute is sharded across both GPUs; arms then run in two waves of
# one-per-GPU (each needs ~8.3 GiB). Resumable like the main driver.
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
LOGDIR=results/r1_recon_probe_hydrant
mkdir -p "$LOGDIR"

# wait for any earlier probe run to release the GPUs
while pgrep -f "r1_recon_probe.py --feature" > /dev/null; do sleep 60; done

for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i $PY experiments/r1_recon_probe.py --precompute-only \
        --split hydrant-full --shard "$i/2" \
        >> "$LOGDIR/precompute_$i.log" 2>&1 &
done
wait

for wave in "dpt mv" "desc mvdesc"; do
    gpu=0
    for feat in $wave; do
        if [ -f "$LOGDIR/$feat/metrics.json" ]; then
            echo "$feat already done, skipping"
            continue
        fi
        CUDA_VISIBLE_DEVICES=$gpu $PY experiments/r1_recon_probe.py \
            --feature "$feat" --split hydrant-full \
            >> "$LOGDIR/train_$feat.log" 2>&1 &
        gpu=$(( (gpu + 1) % 2 ))
    done
    wait
done
echo "r1 hydrant-full done"
