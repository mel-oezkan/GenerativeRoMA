#!/usr/bin/env bash
# Downloads the MegaDepth-1500 and ScanNet-1500 eval sets into shared group
# storage and symlinks them into this repo's data/ dir.
# Idempotent: wget -c resumes partial downloads, existing files are kept.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/visinf/projects_students/dlcv2025_groupZ/romav2_data

mkdir -p "$DATA_ROOT/megadepth" "$DATA_ROOT/scannet/scans"

# symlink repo data/{megadepth,scannet} -> shared storage
for d in megadepth scannet; do
    if [ ! -e "$REPO_DIR/data/$d" ]; then
        ln -s "$DATA_ROOT/$d" "$REPO_DIR/data/$d"
    fi
done

cd "$DATA_ROOT"

# MegaDepth-1500 pair lists
for npz in 0015_0.1_0.3 0015_0.3_0.5 0022_0.1_0.3 0022_0.3_0.5 0022_0.5_0.7; do
    wget -c "https://github.com/Parskatt/storage/releases/download/mega1500/${npz}.npz" -O "megadepth/${npz}.npz"
done

# MegaDepth-1500 images
if [ ! -d megadepth/Undistorted_SfM ]; then
    wget -c https://github.com/Parskatt/storage/releases/download/mega1500/megadepth_test_1500.tar
    tar -xf megadepth_test_1500.tar
    mv megadepth_test_1500/Undistorted_SfM megadepth/
    rmdir megadepth_test_1500
    rm megadepth_test_1500.tar
fi

# ScanNet-1500
wget -c https://github.com/Parskatt/storage/releases/download/scannet1500/test.npz -O scannet/scans/test.npz
if [ ! -d scannet/scans/scans_test ]; then
    wget -c https://github.com/Parskatt/storage/releases/download/scannet1500/scannet_test_1500.tar
    tar -xf scannet_test_1500.tar
    mv scannet_test_1500 scannet/scans/scans_test
    rm scannet_test_1500.tar
fi

echo "Done. Data in $DATA_ROOT, symlinked from $REPO_DIR/data/"
