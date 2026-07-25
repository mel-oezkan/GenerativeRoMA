#!/usr/bin/env bash
# Frozen-DINOv3 control for R2 v1 (docs/R2.md): same data (r3-4cat), same
# losses/steps/accum/GAN recipe as r2_v1, but the mv_vit is replaced by a
# linear 2048->1024 projection — only the projection, the DPT head and
# (joint) the decoder train. The gap to the r2_v1 arms is the contribution
# of the deep trained encoder; the recon side of this baseline is the R1
# desc ceiling by construction.
#
# Waits for the r23 probe pass to free the GPUs, then match (GPU0) +
# joint (GPU1). Resumable as usual.
#
# Recipe: configs/experiment/r2_v1_desc.yaml.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

EXP=r2_v1_desc

log "waiting for the r23 probe pass to finish"
while ! done_if results/r2_v1_probe/match/metrics.json \
   || ! done_if results/r2_v1_probe/joint/metrics.json; do
    sleep 300
done

log "GPUs free"
smoke "$EXP" joint 0

train "$EXP" match 0 &
train "$EXP" joint 1 &
wait
log "r2 v1 desc-baseline done"
