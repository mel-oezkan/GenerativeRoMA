#!/usr/bin/env bash
# R2 post-hoc recon probe (docs/R2.md): measure appearance retained in the
# mv tokens of the R2-trained match vs joint matchers with the R1 RAE probe
# (fresh decoder per arm, hydrant-full split — comparable to stock RoMaV2).
#   0. fill the frame descs missing from the desc cache (single process)
#   1. mv extraction from the two R2 ckpts, one arm per GPU
#   2. probe training, one arm per GPU
# Resumable: extraction skips cached pairs, probe resumes from ckpt.pt.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

RUN=${RUN:-r2_v0}
STEP=${STEP:-6000}
CACHE_NAME=${CACHE_NAME:-r2_probe_320}
FEATS=/visinf/projects_students/dlcv2025_groupZ/romav2_feats
OUT=results/r2_probe
mkdir -p "$OUT"

log "desc fill"
gpu 0 experiments/r2_probe_precompute.py fill_desc=true \
    >> "$OUT/desc_fill.log" 2>&1

log "mv extraction (match GPU0, joint GPU1)"
dev=0
for arm in match joint; do
    gpu "$dev" experiments/r2_probe_precompute.py "run=$RUN" "arm=$arm" \
        "step=$STEP" "cache_name=$CACHE_NAME" \
        >> "$OUT/extract_$arm.log" 2>&1 &
    dev=$((dev + 1))
done
wait

log "probe training (match GPU0, joint GPU1)"
dev=0
for arm in match joint; do
    probe mv hydrant-full "$dev" "$OUT/$arm" \
        "cache_dir=$FEATS/$CACHE_NAME/$arm" &
    dev=$((dev + 1))
done
wait
log "r2 probe done"
