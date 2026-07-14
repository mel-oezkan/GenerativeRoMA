#!/usr/bin/env bash
# R1 b3 arm: RAE probe on mv_vit block-3 output tokens (768ch @ /16), the
# mid-depth tap where the PCA visualization still shows appearance before
# the b5-b9 collapse (see r1_visualize_mvvit). 3cat split, same recipe as
# scripts/r1_driver.sh, so the result slots between desc (19.56 dB) and
# mv (12.09 dB). Precompute writes .b3.pt sidecars for the existing pairs,
# sharded across both GPUs; training runs on GPU 0. Resumable.
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
LOGDIR=results/r1_recon_probe
mkdir -p "$LOGDIR"

while pgrep -f "experiments/r1_recon_probe.py" > /dev/null; do sleep 120; done

for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i $PY experiments/r1_recon_probe.py --precompute-only \
        --shard "$i/2" \
        >> "$LOGDIR/precompute_b3_$i.log" 2>&1 &
done
wait

if [ -f "$LOGDIR/b3/metrics.json" ]; then
    echo "b3 already done, skipping"
else
    CUDA_VISIBLE_DEVICES=0 $PY experiments/r1_recon_probe.py --feature b3 \
        >> "$LOGDIR/train_b3.log" 2>&1
fi
echo "r1 b3 done"
