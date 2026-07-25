#!/usr/bin/env bash
# R1 decoder-scale sweep (docs/R1.md appendix): does decoder capacity change
# the absolute numbers and/or the desc<->b3 gap? Runs the RAE probe at size
# S (21.6M body) and L (303M body) on the desc (ceiling control) and b3 (R2
# candidate) arms, 3cat split, same recipe as the B runs. S fits one GPU ->
# both S runs in parallel; L does not fit 11 GB -> pipelined across both
# GPUs (pipeline=true), so the two L runs go sequentially. Features are
# already cached.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

SPLIT=${SPLIT:-3cat}
LOGDIR=results/r1_recon_probe
mkdir -p "$LOGDIR"

wait_for_pattern "experiments/r1_recon_probe.py"

# phase 1: S runs, one arm per GPU
dev=0
for arm in desc b3; do
    probe "$arm" "$SPLIT" "$dev" "$LOGDIR/${arm}_S" decoder.size=S &
    dev=$((dev + 1))
done
wait

# phase 2: L runs, pipelined across both GPUs, sequential
for arm in desc b3; do
    probe "$arm" "$SPLIT" 0,1 "$LOGDIR/${arm}_L" decoder.size=L pipeline=true
done
log "r1 scale sweep done"
