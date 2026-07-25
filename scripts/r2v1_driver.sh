#!/usr/bin/env bash
# R2 v1: from-scratch matcher training scaled up from the v0 pilot. Same
# r3-4cat data/filters/desc-cache as R3 (census + precompute already done by
# r3_driver.sh — this driver only trains). match (GPU0) + joint (GPU1) in
# parallel; resumable via ckpt.pt; arms with metrics.json are skipped.
#
# Recipe (12000 steps, effective batch 16, delayed GAN on the joint arm):
# configs/experiment/r2_v1.yaml.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

EXP=r2_v1
log "launching $EXP match + joint (from scratch)"
train "$EXP" match 0 &
train "$EXP" joint 1 &
wait
log "r2 v1 done"
