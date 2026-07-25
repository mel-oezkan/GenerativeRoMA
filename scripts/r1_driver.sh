#!/usr/bin/env bash
# R1 recon probe, 3cat split: precompute the shared feature cache (GPU0),
# then train one decoder per feature arm, alternating GPUs. Arms with a
# metrics.json are skipped, so the script is safe to re-run after adding
# arms. Resumable: re-run after interruption (cache skips done pairs, ckpt
# resumes).
#
# Recipe: configs/r1_probe.yaml.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

SPLIT=${SPLIT:-3cat}
LOGDIR=results/r1_recon_probe
mkdir -p "$LOGDIR"

log "precompute (GPU0)"
gpu 0 experiments/r1_recon_probe.py precompute_only=true "split=$SPLIT" \
    2>&1 | tee -a "$LOGDIR/precompute.log"

# probe v3 (RAE recipe, ViT-B decoder + DINO disc + VGG LPIPS) needs a full
# 1080 Ti per arm -> two waves of one-arm-per-GPU instead of 2 per GPU
for wave in "dpt mv" "desc mvdesc"; do
    dev=0
    for feat in $wave; do
        probe "$feat" "$SPLIT" "$dev" "$LOGDIR/$feat" &
        dev=$(( (dev + 1) % 2 ))
    done
    wait
done
log "r1 done"
