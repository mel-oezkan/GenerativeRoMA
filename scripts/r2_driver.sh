#!/usr/bin/env bash
# R2 v0 pipeline (docs/R2.md): waits for the running R1 scale sweep to
# release both GPUs, then
#   1. per-frame desc precompute, one shard per GPU (~13k frames)
#   2. 30-step GPU smoke run of the joint arm (end-to-end check of cache +
#      dataloader + model before the long runs; aborts on failure)
#   3. real runs: match (GPU0) + joint (GPU1) in parallel, then recon (GPU0)
# Resumable: precompute skips cached frames, train resumes from ckpt.pt,
# arms with metrics.json are skipped entirely.
#
# Recipe: configs/experiment/r2_v0.yaml.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

EXP=r2_v0
SPLIT=hydrant-full
OUT=results/$EXP
mkdir -p "$OUT"

log "waiting for R1 scale sweep to finish"
wait_for_pattern "r1_scale_driver.sh|experiments/r1_recon_probe.py" 300
log "GPUs free, starting desc precompute"

for i in 0 1; do
    gpu "$i" experiments/r2_precompute_desc.py "split=$SPLIT" "shard=$i/2" \
        >> "$OUT/precompute$i.log" 2>&1 &
done
wait
log "precompute done"

smoke "$EXP" joint 0

train "$EXP" match 0 &
train "$EXP" joint 1 &
wait
log "match+joint done, launching recon control"

train "$EXP" recon 0
log "r2 v0 pipeline done"
