#!/usr/bin/env bash
# R2 post-hoc recon probe (docs/R2.md): measure appearance retained in the
# mv tokens of the R2-trained match vs joint matchers with the R1 v3 RAE
# probe (fresh decoder per arm, hydrant-full split — comparable to stock
# RoMaV2 mv 13.29 dB / desc 18.95 dB).
#   0. fill the 393 frame descs missing from r2_desc_320 (single process)
#   1. mv extraction from the two R2 ckpts, one arm per GPU
#   2. probe training, one arm per GPU
# Resumable: extraction skips cached pairs, probe resumes from ckpt.pt.
set -euo pipefail

cd /visinf/home/lab_mozkan/GenerativeRoMA
PY=~/miniconda3/envs/cv/bin/python
OUT=results/r2_probe
CACHE=/visinf/projects_students/dlcv2025_groupZ/romav2_feats/r2_probe_320
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[driver $(date +%H:%M)] desc fill"
CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_probe_precompute.py --fill-desc \
    >> "$OUT/desc_fill.log" 2>&1

echo "[driver $(date +%H:%M)] mv extraction (match GPU0, joint GPU1)"
CUDA_VISIBLE_DEVICES=0 $PY experiments/r2_probe_precompute.py --arm match \
    >> "$OUT/extract_match.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY experiments/r2_probe_precompute.py --arm joint \
    >> "$OUT/extract_joint.log" 2>&1 &
wait

echo "[driver $(date +%H:%M)] probe training (match GPU0, joint GPU1)"
if [ ! -f "$OUT/match/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY experiments/r1_recon_probe.py --feature mv \
        --split hydrant-full --cache-dir "$CACHE/match" --out-dir "$OUT/match" \
        >> "$OUT/train_match.log" 2>&1 &
fi
if [ ! -f "$OUT/joint/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=1 $PY experiments/r1_recon_probe.py --feature mv \
        --split hydrant-full --cache-dir "$CACHE/joint" --out-dir "$OUT/joint" \
        >> "$OUT/train_joint.log" 2>&1 &
fi
wait
echo "[driver $(date +%H:%M)] r2 probe done"
