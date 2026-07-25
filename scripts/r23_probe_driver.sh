#!/usr/bin/env bash
# Post-hoc measurement pass over the R3 fine-tune and R2 v1 checkpoints
# (docs/R3.md, docs/R2.md "v1"):
#   0. per-category matching eval for r3_ft match/joint, the pretrained
#      zero-shot baseline, and r2_v1 match/joint
#   1. desc fill for the hydrant-full probe pairs (no-op if the desc cache
#      is already complete from the R2 v0 probe pass)
#   2. mv extraction from all 4 ckpts (hydrant-full probe pairs, so PSNR
#      stays comparable to stock RoMaV2 / the desc ceiling)
#   3. R1 RAE probe per arm (fresh RAEDecoder-B), 2 arms per wave
# Resumable: extraction skips cached pairs, probes resume from ckpt.pt,
# finished probes (metrics.json) are skipped.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

FEATS=/visinf/projects_students/dlcv2025_groupZ/romav2_feats
# run:step:cache-name:probe-results-dir
RUNS=(
  "r3_ft:6000:r3_probe_320:results/r3_probe"
  "r2_v1:12000:r2v1_probe_320:results/r2_v1_probe"
)

log "per-category evals"
mkdir -p results/r3_probe results/r2_v1_probe
if ! done_if results/r3_ft/pretrained/metrics_per_cat.json; then
    gpu 0 experiments/r3_eval_cats.py run=r3_ft arm=pretrained \
        >> results/r3_ft/eval_cats.log 2>&1
fi
for arm in match joint; do
    dev=0
    for spec in "${RUNS[@]}"; do
        IFS=: read -r run step _ _ <<< "$spec"
        if ! done_if "results/$run/$arm/metrics_per_cat.json"; then
            gpu "$dev" experiments/r3_eval_cats.py "run=$run" "arm=$arm" \
                "step=$step" >> "results/$run/eval_cats.log" 2>&1 &
        fi
        dev=$((dev + 1))
    done
    wait
done

log "desc fill"
gpu 0 experiments/r2_probe_precompute.py fill_desc=true \
    >> results/r3_probe/desc_fill.log 2>&1

for spec in "${RUNS[@]}"; do
    IFS=: read -r run step cache out <<< "$spec"
    log "mv extraction $run (match GPU0, joint GPU1)"
    dev=0
    for arm in match joint; do
        gpu "$dev" experiments/r2_probe_precompute.py "run=$run" "arm=$arm" \
            "step=$step" "cache_name=$cache" >> "$out/extract_$arm.log" 2>&1 &
        dev=$((dev + 1))
    done
    wait
done

for spec in "${RUNS[@]}"; do
    IFS=: read -r run _ cache out <<< "$spec"
    log "probe wave: $run (match GPU0, joint GPU1)"
    dev=0
    for arm in match joint; do
        probe mv hydrant-full "$dev" "$out/$arm" \
            "cache_dir=$FEATS/$cache/$arm" &
        dev=$((dev + 1))
    done
    wait
done
log "r23 probe pass done"
