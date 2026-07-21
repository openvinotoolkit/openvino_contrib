#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Activate conda environment (optional, uncomment if needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ov_sam6d

# Get project root directory (git repo root)
export PROJECT_ROOT=$(git rev-parse --show-toplevel)

# Set data and output paths
export CAD_PATH=$PROJECT_ROOT/modules/3d/OV-SAM-6D/SAM-6D/Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=$PROJECT_ROOT/modules/3d/OV-SAM-6D/SAM-6D/Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=$PROJECT_ROOT/modules/3d/OV-SAM-6D/SAM-6D/Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=$PROJECT_ROOT/modules/3d/OV-SAM-6D/SAM-6D/Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=$PROJECT_ROOT/modules/3d/OV-SAM-6D/SAM-6D/Data/Example/outputs

# Set instance segmentation model
export SEGMENTOR_MODEL=sam
# Set pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

# Set CUDA environment variables
#export PATH=/usr/local/cuda-12.4/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Check dependencies
command -v git >/dev/null 2>&1 || { echo >&2 "[ERROR] git is not installed. Please install git."; exit 1; }
command -v conda >/dev/null 2>&1 || echo "[WARNING] conda not found. Please activate your conda environment manually."
#command -v nvcc >/dev/null 2>&1 || echo "[WARNING] Please check your CUDA installation."

# Check key files and directories
for f in "$CAD_PATH" "$RGB_PATH" "$DEPTH_PATH" "$CAMERA_PATH"; do
    if [ ! -f "$f" ]; then
        echo "[WARNING] File not found: $f"
    fi
done
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "[INFO] Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

echo "[INFO] Environment variables set."
echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] CAD_PATH=$CAD_PATH"
echo "[INFO] RGB_PATH=$RGB_PATH"
echo "[INFO] DEPTH_PATH=$DEPTH_PATH"
echo "[INFO] CAMERA_PATH=$CAMERA_PATH"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
#echo "[INFO] CUDA PATH=$PATH"
#echo "[INFO] CUDA LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
