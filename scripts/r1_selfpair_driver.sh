#!/usr/bin/env bash
# R1 self-pair control: mv features with the matcher conditioned on the
# anchor itself (A == B). Same 3cat split/recipe as scripts/r1_driver.sh;
# eval anchors coincide with the normal mv arm, so the result is directly
# comparable to results/r1_recon_probe/mv. Tests whether cross-view
# conditioning destroys appearance, or the matcher stages do it even on
# identical inputs.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

SPLIT=${SPLIT:-3cat}
LOGDIR=results/r1_recon_probe
mkdir -p "$LOGDIR"

# wait for the hydrant driver and any probe process to release the GPUs
wait_for_pattern "r1_hydrant_driver.sh|experiments/r1_recon_probe.py"

log "self-pair precompute, sharded across both GPUs"
for i in 0 1; do
    gpu "$i" experiments/r1_recon_probe.py precompute_only=true \
        "split=$SPLIT" self_pair=true "shard=$i/2" \
        >> "$LOGDIR/precompute_selfpair_$i.log" 2>&1 &
done
wait

probe mv "$SPLIT" 0 "$LOGDIR/mv_selfpair" self_pair=true
log "r1 mv self-pair done"
