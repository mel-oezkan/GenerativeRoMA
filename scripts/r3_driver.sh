#!/usr/bin/env bash
# R3 pipeline: fine-tune the PRETRAINED RoMaV2 matcher on 4 CO3D categories
# (hydrant, bench, toybus, toytruck) with the R2 joint matching+recon recipe.
#   0. wait for the co3d_full download (download_bench_toybus_toytruck.sh),
#      then extract all bench/toybus/toytruck zips (stamped, resumable)
#   1. covis census (CPU; reuses the hydrant census for shared pairs)
#      -> the FULL filter pipeline exists before any caching
#   2. per-frame desc precompute on the *filtered* pairs, one shard per GPU
#      (hydrant frames already cached from R2 are skipped)
#   3. 30-step smoke run of joint-ft (abort on failure)
#   4. real runs: match-ft (GPU0) + joint-ft (GPU1)
# Resumable everywhere: extraction stamps, census/precompute skip cached,
# train resumes from ckpt.pt, arms with metrics.json are skipped.
#
# Recipe (pretrained init, matcher LR 2e-5): configs/experiment/r3_ft.yaml.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

EXP=r3_ft
SPLIT=r3-4cat
DATA=/visinf/projects_students/dlcv2025_groupZ/co3d_full
OUT=results/$EXP
mkdir -p "$OUT"

log "waiting for the co3d download to finish"
while ! grep -q "download phase done, FAILED=0" "$DATA/download_btt.log" 2>/dev/null; do
    if grep -q "download phase done, FAILED=1" "$DATA/download_btt.log" 2>/dev/null; then
        log "DOWNLOAD FAILED (see $DATA/download_btt.log), aborting"
        exit 1
    fi
    sleep 120
done

log "download done, extracting zips"
mkdir -p "$DATA/.extract_stamps"
for z in "$DATA"/bench_*.zip "$DATA"/toybus_*.zip "$DATA"/toytruck_*.zip; do
    stamp="$DATA/.extract_stamps/$(basename "$z").done"
    if [ ! -f "$stamp" ]; then
        log "unzip $(basename "$z")"
        unzip -o -q "$z" -d "$DATA"
        touch "$stamp"
    fi
done
log "extraction done, covis census (CPU)"

$PY experiments/r2_pair_covis.py "split=$SPLIT" >> "$OUT/census.log" 2>&1
log "census done, desc precompute (filtered pairs)"

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
log "r3 fine-tune pipeline done"
