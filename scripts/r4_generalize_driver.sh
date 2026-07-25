#!/usr/bin/env bash
# R4 generalization sweep (docs/R4.md).
#
# Tier 1 (GPU 0): category transfer -- the co3d-unseen split, 17 CO3D
#   categories disjoint from every training category. Same protocol/metrics
#   as the in-domain r3-4cat eval, so the two are directly comparable per arm.
# Tier 2 (GPU 1): regime transfer -- MegaDepth-1500 two-view pose AUC at the
#   turbo setting (320 px, no refiners: our training resolution, and no arm
#   of ours has refiners). The pretrained checkpoint is run both ways as the
#   anchor.
#
# Both tiers are resumable: every step skips if its output json exists.
# Launch detached:  nohup bash scripts/r4_generalize_driver.sh &
set -u
source "$(dirname "$0")/lib.sh"

mkdir -p results/r4_logs

# run arm step desc_only
ARMS=(
  "r2_v1 match 12000 false"
  "r2_v1 joint 12000 false"
  "r2_v2 joint 12000 false"
  "r2_v1_desc match 12000 true"
  "r2_v1_desc joint 12000 true"
  "r3_ft match 6000 false"
  "r3_ft joint 6000 false"
  "r3_ft pretrained 0 false"
  "r3_ft pretrained-refined 0 false"
)

tier1() {
  for spec in "${ARMS[@]}"; do
    set -- $spec
    local run=$1 arm=$2 step=$3 desc=$4
    done_if "results/$run/$arm/metrics_per_cat_co3d-unseen.json" \
      && { log "skip tier1 $run/$arm (done)"; continue; }
    log "tier1 $run/$arm"
    gpu 0 experiments/r3_eval_cats.py "run=$run" "arm=$arm" "step=$step" \
      split=co3d-unseen tag=co3d-unseen "model.desc_only=$desc" \
      >> "results/r4_logs/tier1_${run}_${arm}.log" 2>&1 \
      || log "FAILED tier1 $run/$arm"
  done
  log "tier1 done"
}

tier2() {
  for spec in "${ARMS[@]}"; do
    set -- $spec
    local run=$1 arm=$2 step=$3 desc=$4
    [ "$arm" = "pretrained-refined" ] && continue  # handled via model.refiners
    done_if "results/$run/$arm/bench_mega1500.json" \
      && { log "skip tier2 $run/$arm (done)"; continue; }
    log "tier2 $run/$arm"
    gpu 1 experiments/r3_bench_pose.py "run=$run" "arm=$arm" "step=$step" \
      benchmark=mega1500 "model.desc_only=$desc" \
      >> "results/r4_logs/tier2_${run}_${arm}.log" 2>&1 \
      || log "FAILED tier2 $run/$arm"
  done
  # anchor: the released model with its refiners, same 320 px setting
  if ! done_if results/r3_ft/pretrained/bench_mega1500_refined.json; then
    log "tier2 pretrained+refiners"
    gpu 1 experiments/r3_bench_pose.py run=r3_ft arm=pretrained \
      benchmark=mega1500 model.refiners=true \
      >> results/r4_logs/tier2_pretrained_refined.log 2>&1 \
      || log "FAILED tier2 pretrained+refiners"
  fi
  log "tier2 done"
}

# usage: r4_generalize_driver.sh [tier1|tier2|both]   (default both)
WHICH=${1:-both}
log "R4 driver start ($WHICH)"
PIDS=()
[ "$WHICH" = "both" -o "$WHICH" = "tier1" ] && { tier1 & PIDS+=($!); }
[ "$WHICH" = "both" -o "$WHICH" = "tier2" ] && { tier2 & PIDS+=($!); }
wait "${PIDS[@]}"
log "R4 driver done"
