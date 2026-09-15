#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../Data/BOP}"
DATASET="${DATASET:-lmo}"
CONDA_ENV="${CONDA_ENV:-ov_sam6d}"
OV_DEVICE="${OV_DEVICE:-GPU}"
SEGMENTOR_MODEL="${SEGMENTOR_MODEL:-fastsam}"
MAX_IMAGES="${MAX_IMAGES:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_NAME="${OUTPUT_NAME:-subset_eval_ov}"
DETECTION_PATH="${DETECTION_PATH:-}"

DATASET_DIR="$DATA_ROOT/$DATASET"
ISM_RESULTS_DIR="$DATASET_DIR/bop/ism_ov_${OV_DEVICE,,}_${SEGMENTOR_MODEL}_results"

log() {
  printf '[run_pos] %s\n' "$*"
}

die() {
  printf '[run_pos] ERROR: %s\n' "$*" >&2
  exit 1
}

check_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

main() {
  check_cmd "$PYTHON_BIN"
  check_cmd conda

  [[ -d "$SCRIPT_DIR" ]] || die "Invalid script directory: $SCRIPT_DIR"
  [[ -d "$DATASET_DIR" ]] || die "Dataset dir not found: $DATASET_DIR"

  if [[ -z "$DETECTION_PATH" ]]; then
    local merged_json=""
    local compat_json=""
    if [[ -n "$MAX_IMAGES" ]]; then
      merged_json="$ISM_RESULTS_DIR/result_${DATASET}_${MAX_IMAGES}imgs.json"
      compat_json="$SCRIPT_DIR/../Instance_Segmentation_Model/log/sam_ov/result_${DATASET}_${MAX_IMAGES}imgs.json"
      if [[ -f "$merged_json" ]]; then
        DETECTION_PATH="$merged_json"
      elif [[ -f "$compat_json" ]]; then
        DETECTION_PATH="$compat_json"
      fi
    fi

    if [[ -z "$DETECTION_PATH" ]]; then
      local merged_glob="$ISM_RESULTS_DIR/result_${DATASET}_*imgs.json"
      local compat_glob="$SCRIPT_DIR/../Instance_Segmentation_Model/log/sam_ov/result_${DATASET}_*imgs.json"
      if compgen -G "$merged_glob" > /dev/null; then
        DETECTION_PATH="$(ls -1t $merged_glob | head -n 1)"
      elif compgen -G "$compat_glob" > /dev/null; then
        DETECTION_PATH="$(ls -1t $compat_glob | head -n 1)"
      fi
    fi

    if [[ -z "$DETECTION_PATH" ]]; then
      die "Detection JSON not found. Run Instance_Segmentation_Model/run_ism.sh first or set DETECTION_PATH"
    fi
  fi

  [[ -f "$DETECTION_PATH" ]] || die "Detection JSON not found: $DETECTION_PATH"

  if [[ -z "$MAX_IMAGES" ]]; then
    local resolved_max=""
    local meta_from_detection="${DETECTION_PATH%.json}.meta.json"
    local meta_from_results="$ISM_RESULTS_DIR/result_${DATASET}_*imgs.meta.json"

    if [[ -f "$meta_from_detection" ]]; then
      resolved_max="$($PYTHON_BIN - <<'PY' "$meta_from_detection"
import json, sys
with open(sys.argv[1], 'r') as f:
    meta = json.load(f)
print(int(meta.get('max_images_requested', meta.get('n_detection_images', 0))))
PY
)"
    elif compgen -G "$meta_from_results" > /dev/null; then
      local latest_meta
      latest_meta="$(ls -1t $meta_from_results | head -n 1)"
      resolved_max="$($PYTHON_BIN - <<'PY' "$latest_meta"
import json, sys
with open(sys.argv[1], 'r') as f:
    meta = json.load(f)
print(int(meta.get('max_images_requested', meta.get('n_detection_images', 0))))
PY
)"
    elif [[ "$DETECTION_PATH" =~ _([0-9]+)imgs\.json$ ]]; then
      resolved_max="${BASH_REMATCH[1]}"
    fi

    if [[ -z "$resolved_max" || "$resolved_max" == "0" ]]; then
      resolved_max="10"
      log "Could not auto-resolve max samples, defaulting to $resolved_max"
    fi
    MAX_IMAGES="$resolved_max"
  fi

  log "Starting PEM subset eval"
  log "Dataset: $DATASET"
  log "OV device: $OV_DEVICE"
  log "Max samples: $MAX_IMAGES"
  log "Detection path: $DETECTION_PATH"

  cd "$SCRIPT_DIR"
  conda run --no-capture-output -n "$CONDA_ENV" \
    "$PYTHON_BIN" test_bop_subset_eval_ov.py \
    --dataset "$DATASET" \
    --device "$OV_DEVICE" \
    --max_samples "$MAX_IMAGES" \
    --detection_path "$DETECTION_PATH" \
    --output_name "$OUTPUT_NAME"
}

main "$@"
