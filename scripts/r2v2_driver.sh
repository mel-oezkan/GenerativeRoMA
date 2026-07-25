#!/usr/bin/env bash
# R2 v2: same as the r2_v1 joint run, ONLY change = reconstruct BOTH views.
# The train-time recon loss decodes mv_A -> image A and mv_B -> image B and
# *averages* the two (L1 + LPIPS + GAN), so the recon-vs-matching weight is
# identical to v1 -- the sole difference is that view B's mv tokens now get
# recon gradient too. See docs/R2.md.
#
# Only the joint arm is (re)trained; the match baseline is r2_v1/match.
# The batch/accum split that keeps this inside 11 GB is in the config.
#
# Recipe: configs/experiment/r2_v2.yaml.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

EXP=r2_v2
GPU=${GPU:-1}
log "launching $EXP joint (two-view recon) on GPU $GPU"
train "$EXP" joint "$GPU"
log "r2 v2 done"
