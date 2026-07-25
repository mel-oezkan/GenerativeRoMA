#!/usr/bin/env bash
# R1 b3 arm: RAE probe on mv_vit block-3 output tokens (768ch @ /16), the
# mid-depth tap where the PCA visualization still shows appearance before
# the b5-b9 collapse (see r1_visualize_mvvit). 3cat split, same recipe as
# scripts/r1_driver.sh, so the result slots between desc and mv. Precompute
# writes .b3.pt sidecars for the existing pairs, sharded across both GPUs;
# training runs on GPU 0.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

SPLIT=${SPLIT:-3cat}
LOGDIR=results/r1_recon_probe
mkdir -p "$LOGDIR"

wait_for_pattern "experiments/r1_recon_probe.py"

log "b3 sidecar precompute, sharded across both GPUs"
for i in 0 1; do
    gpu "$i" experiments/r1_recon_probe.py precompute_only=true \
        "split=$SPLIT" "shard=$i/2" >> "$LOGDIR/precompute_b3_$i.log" 2>&1 &
done
wait

probe b3 "$SPLIT" 0 "$LOGDIR/b3"
log "r1 b3 done"
