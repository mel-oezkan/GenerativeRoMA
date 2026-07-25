#!/usr/bin/env bash
# R1 hydrant-full: same probe/recipe as scripts/r1_driver.sh but on all 726
# extracted hydrant sequences (10 train pairs/seq; eval = the same 6 hydrant
# sequences as the 3cat split). Precompute is sharded across both GPUs; arms
# then run in two waves of one-per-GPU (each needs ~8.3 GiB).
set -euo pipefail
source "$(dirname "$0")/lib.sh"

SPLIT=hydrant-full
LOGDIR=results/r1_recon_probe_hydrant
mkdir -p "$LOGDIR"

# wait for any earlier probe run to release the GPUs
wait_for_pattern "r1_recon_probe.py feature=" 60

log "precompute, sharded across both GPUs"
for i in 0 1; do
    gpu "$i" experiments/r1_recon_probe.py precompute_only=true \
        "split=$SPLIT" "shard=$i/2" >> "$LOGDIR/precompute_$i.log" 2>&1 &
done
wait

for wave in "dpt mv" "desc mvdesc"; do
    dev=0
    for feat in $wave; do
        probe "$feat" "$SPLIT" "$dev" "$LOGDIR/$feat" &
        dev=$(( (dev + 1) % 2 ))
    done
    wait
done
log "r1 hydrant-full done"
