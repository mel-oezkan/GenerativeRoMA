#!/usr/bin/env bash
# Shared driver helpers. Source this from every scripts/*_driver.sh:
#
#     source "$(dirname "$0")/lib.sh"
#
# Drivers orchestrate (which arm, which GPU, in what order, what to wait
# for); *hyperparameters live in configs/*, never here* — a driver passes
# `experiment=<name> arm=<arm>` and nothing else, so the config file is the
# single description of the run.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY=${PY:-~/miniconda3/envs/cv/bin/python}
export PYTHONPATH="$REPO"
# fp32 training sits near the 11 GB ceiling; avoid fragmentation OOMs
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

log() { echo "[$(basename "$0" .sh) $(date +%H:%M)] $*"; }

# gpu <n> <script> [hydra overrides...] — run one job on one GPU
gpu() {
    local dev=$1 script=$2
    shift 2
    CUDA_VISIBLE_DEVICES=$dev $PY "$REPO/$script" "$@"
}

# done_if <file> — true when a stage's output already exists (drivers are
# re-runnable: finished stages are skipped, unfinished ones resume from
# their own checkpoints)
done_if() { [ -f "$1" ]; }

# wait_for_pattern <pgrep pattern> [seconds] — block until no process
# matches, so a driver can queue behind whatever is holding the GPUs
wait_for_pattern() {
    local pat=$1 interval=${2:-120}
    while pgrep -f "$pat" > /dev/null; do sleep "$interval"; done
}

# train <run-config> <arm> <gpu> [extra overrides...]
# Skips arms that already have metrics.json; otherwise resumes from ckpt.pt.
train() {
    local experiment=$1 arm=$2 dev=$3
    shift 3
    local out="results/$experiment"
    mkdir -p "$out"
    if done_if "$out/$arm/metrics.json"; then
        log "$experiment/$arm already done, skipping"
        return 0
    fi
    log "train $experiment/$arm on GPU $dev"
    gpu "$dev" experiments/r2_train.py "experiment=$experiment" "arm=$arm" "$@" \
        >> "$out/train_$arm.log" 2>&1
}

# smoke <experiment> <arm> <gpu> [extra overrides...]
# Short end-to-end run into results/<experiment>_smoke; aborts the driver on
# failure, so a broken cache/dataloader/model is caught before the long runs.
smoke() {
    local experiment=$1 arm=$2 dev=$3
    shift 3
    local out="results/$experiment"
    mkdir -p "$out"
    log "smoke run ($experiment/$arm, 30 steps)"
    gpu "$dev" experiments/r2_train.py "experiment=$experiment" "arm=$arm" \
        optim.steps=30 "out_dir=results/${experiment}_smoke" "$@" \
        >> "$out/smoke.log" 2>&1 \
        || { log "SMOKE RUN FAILED, aborting (see $out/smoke.log)"; exit 1; }
    log "smoke OK"
}

# probe <feature> <split> <gpu> <out-dir> [extra overrides...]
# R1 RAE probe; skips arms that already have metrics.json.
probe() {
    local feature=$1 split=$2 dev=$3 out=$4
    shift 4
    if done_if "$out/metrics.json"; then
        log "probe $out already done, skipping"
        return 0
    fi
    mkdir -p "$(dirname "$out")"
    log "probe $feature -> $out on GPU $dev"
    gpu "$dev" experiments/r1_recon_probe.py "feature=$feature" "split=$split" \
        "out_dir=$out" "$@" >> "$(dirname "$out")/train_$(basename "$out").log" 2>&1
}
