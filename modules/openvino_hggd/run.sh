#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# HGGD Intel iGPU — Run inference on one scene and report FPS + Accuracy
#
# Usage: bash run.sh <CHECKPOINT_PATH> <DATASET_PATH> <SCENE_PATH> [SCENE_ID] [OV_DEVICE]
#
# Example:
#   bash run.sh \
#     /path/to/HGGD_realsense_checkpoint \
#     /path/to/dataset/6dto2drefine_realsense \
#     /path/to/graspnet \
#     100 GPU

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT_PATH="${1:?Usage: bash run.sh <checkpoint> <dataset> <scenes> [scene_id] [ov_device]}"
DATASET_PATH="${2:?Missing dataset path}"
SCENE_PATH="${3:?Missing scene path}"
SCENE_ID="${4:-100}"
OV_DEVICE="${5:-GPU}"
SCENE_END=$((SCENE_ID + 1))

OUTPUT_DIR="$SCRIPT_DIR/output/scene_${SCENE_ID}"
DUMP_DIR="$OUTPUT_DIR/pred"

echo "=== HGGD Intel iGPU Inference ==="
echo "Device:     $OV_DEVICE"
echo "Scene:      $SCENE_ID"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output:     $OUTPUT_DIR"
echo ""

mkdir -p "$DUMP_DIR"

python "$SCRIPT_DIR/infer.py" \
    --ov-device "$OV_DEVICE" \
    --precision-hint default \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --dataset-path "$DATASET_PATH" \
    --scene-path "$SCENE_PATH" \
    --scene-l "$SCENE_ID" --scene-r "$SCENE_END" \
    --input-h 360 --input-w 640 \
    --anchor-num 7 --all-points-num 25600 \
    --center-num 48 --group-num 512 \
    --dump-dir "$DUMP_DIR"

echo ""
echo "=== Done ==="
echo "Predictions: $DUMP_DIR"
echo "Log:         $OUTPUT_DIR/inference.log"
