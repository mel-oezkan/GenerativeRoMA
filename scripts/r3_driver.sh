#!/usr/bin/env bash
# R3 pipeline: fine-tune the PRETRAINED RoMaV2 matcher on 4 CO3D categories
# (hydrant, bench, toybus, toytruck) with the R2 joint matching+recon recipe.
#   0. wait for the co3d_full download (download_bench_toybus_toytruck.sh),
#      then extract all bench/toybus/toytruck zips (stamped, resumable)
#   1. covis census (CPU; reuses the hydrant census for shared pairs)
#      -> the FULL filter pipeline exists before any caching
#   2. per-frame desc precompute on the *filtered* pairs, one shard per GPU
#      (hydrant frames already cached from R2 are skipped)
#   3. 30-step smoke run of joint-ft into results/r3_smoke (abort on failure)
#   4. real runs: match-ft (GPU0) + joint-ft (GPU1), init=pretrained,
#      matcher LR 2e-5 (decoder keeps 2e-4), out results/r3_ft
# Resumable everywhere: extraction stamps, census/precompute skip cached,
# train resumes from ckpt.pt, arms with metrics.json are skipped.
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
DATA=/visinf/projects_students/dlcv2025_groupZ/co3d_full
OUT=results/r3_ft
SPLIT=r3-4cat
MATCHER_LR=2e-5
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[driver $(date +%H:%M)] waiting for the co3d download to finish"
while ! grep -q "download phase done, FAILED=0" "$DATA/download_btt.log" 2>/dev/null; do
    if grep -q "download phase done, FAILED=1" "$DATA/download_btt.log" 2>/dev/null; then
        echo "[driver] DOWNLOAD FAILED (see $DATA/download_btt.log), aborting"
        exit 1
    fi
    sleep 120
done

echo "[driver $(date +%H:%M)] download done, extracting zips"
mkdir -p "$DATA/.extract_stamps"
for z in "$DATA"/bench_*.zip "$DATA"/toybus_*.zip "$DATA"/toytruck_*.zip; do
    stamp="$DATA/.extract_stamps/$(basename "$z").done"
    if [ ! -f "$stamp" ]; then
        echo "[driver $(date +%H:%M)] unzip $(basename "$z")"
        unzip -o -q "$z" -d "$DATA"
        touch "$stamp"
    fi
done
echo "[driver $(date +%H:%M)] extraction done, covis census (CPU)"

$PY experiments/r2_pair_covis.py --split $SPLIT >> "$OUT/census.log" 2>&1
echo "[driver $(date +%H:%M)] census done, desc precompute (filtered pairs)"

CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_precompute_desc.py --split $SPLIT \
    --shard 0/2 >> "$OUT/precompute0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY experiments/r2_precompute_desc.py --split $SPLIT \
    --shard 1/2 >> "$OUT/precompute1.log" 2>&1 &
wait
echo "[driver $(date +%H:%M)] precompute done, smoke run"

CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_train.py --arm joint --steps 30 \
    --split $SPLIT --init pretrained --matcher-lr $MATCHER_LR \
    --out-dir results/r3_smoke >> "$OUT/smoke.log" 2>&1 \
    || { echo "[driver] SMOKE RUN FAILED, aborting (see $OUT/smoke.log)"; exit 1; }
echo "[driver $(date +%H:%M)] smoke OK, launching match-ft + joint-ft"

if [ ! -f "$OUT/match/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_train.py --arm match \
        --split $SPLIT --init pretrained --matcher-lr $MATCHER_LR \
        --out-dir "$OUT" >> "$OUT/train_match.log" 2>&1 &
fi
if [ ! -f "$OUT/joint/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=1 $PY experiments/r2_train.py --arm joint \
        --split $SPLIT --init pretrained --matcher-lr $MATCHER_LR \
        --out-dir "$OUT" >> "$OUT/train_joint.log" 2>&1 &
fi
wait
echo "[driver $(date +%H:%M)] r3 fine-tune pipeline done"
