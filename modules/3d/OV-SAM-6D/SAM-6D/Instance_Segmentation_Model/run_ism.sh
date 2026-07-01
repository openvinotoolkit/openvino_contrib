#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Run OpenVINO ISM evaluation on BOP LM-O (10 images) with setup checks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../Data/BOP}"
DATASET="${DATASET:-lmo}"
CONDA_ENV="${CONDA_ENV:-ov_sam6d}"
OV_DEVICE="${OV_DEVICE:-GPU}"
OV_SAM_DEVICE="${OV_SAM_DEVICE:-GPU}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET_DIR="$DATA_ROOT/$DATASET"
CACHE_DIR="$DATA_ROOT/templates_pyrender/$DATASET"
OV_MODEL_DIR="$SCRIPT_DIR/checkpoints/ov_models"

log() {
  printf '[run_ism] %s\n' "$*"
}

die() {
  printf '[run_ism] ERROR: %s\n' "$*" >&2
  exit 1
}

check_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

repair_nested_layout_if_needed() {
  # Some extractions create <dataset>/<dataset>/...; link expected dirs if needed.
  local nested_root="$DATASET_DIR/$DATASET"
  local d
  for d in models test train_pbr eval_output; do
    if [[ ! -e "$DATASET_DIR/$d" && -e "$nested_root/$d" ]]; then
      ln -s "$nested_root/$d" "$DATASET_DIR/$d"
      log "Linked missing $d -> $nested_root/$d"
    fi
  done
  if [[ ! -e "$DATASET_DIR/test_metaData.json" && -e "$nested_root/test_metaData.json" ]]; then
    ln -s "$nested_root/test_metaData.json" "$DATASET_DIR/test_metaData.json"
    log "Linked missing test_metaData.json from nested layout"
  fi
}

main() {
  check_cmd "$PYTHON_BIN"
  check_cmd conda

  [[ -d "$SCRIPT_DIR" ]] || die "Invalid script directory: $SCRIPT_DIR"
  [[ -d "$DATA_ROOT" ]] || die "BOP root not found: $DATA_ROOT"
  [[ -d "$DATASET_DIR" ]] || die "Dataset dir not found: $DATASET_DIR"

  repair_nested_layout_if_needed

  # Validate required LM-O layout used by run_inference_ov_10.py
  [[ -d "$DATASET_DIR/models" ]] || die "Missing $DATASET_DIR/models. Extract the lmo_models.zip in Data/BOP/lmo/."
  [[ -d "$DATASET_DIR/test" ]] || die "Missing $DATASET_DIR/test"
  [[ -d "$DATASET_DIR/train_pbr" ]] || die "Missing $DATASET_DIR/train_pbr. Extract the lmo_train.zip and rename the train dir to train_pbr in Data/BOP/lmo/."
  #[[ -d "$DATASET_DIR/eval_output" ]] || die "Missing $DATASET_DIR/eval_output"
  #[[ -f "$DATASET_DIR/test_metaData.json" ]] || die "Missing $DATASET_DIR/test_metaData.json"

  # Ensure template cache location expected by configs/data/bop.yaml exists.
  mkdir -p "$CACHE_DIR"
  log "Ensured template cache directory: $CACHE_DIR"

  # Validate OV IR files used by infer_ism_ov.py
  [[ -f "$OV_MODEL_DIR/sam_image_encoder.xml" ]] || die "Missing $OV_MODEL_DIR/sam_image_encoder.xml"
  [[ -f "$OV_MODEL_DIR/sam_mask_decoder.xml" ]] || die "Missing $OV_MODEL_DIR/sam_mask_decoder.xml"
  [[ -f "$OV_MODEL_DIR/dinov2_vitl14.xml" ]] || die "Missing $OV_MODEL_DIR/dinov2_vitl14.xml"

  cd "$SCRIPT_DIR"

  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local run_log="log/sam_ov/run_ism_${DATASET}_${stamp}.log"
  mkdir -p "$(dirname "$run_log")"

  log "Starting OpenVINO ISM run"
  log "Dataset: $DATASET"
  log "Conda env: $CONDA_ENV"
  log "OV_DEVICE=$OV_DEVICE OV_SAM_DEVICE=$OV_SAM_DEVICE"
  log "Log file: $run_log"

  # Run inference in background, write output to log file directly
  # so the log is always written even if stdout is piped through head/grep.
  local rc=0
  OV_DEVICE="$OV_DEVICE" \
  OV_SAM_DEVICE="$OV_SAM_DEVICE" \
  PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n "$CONDA_ENV" \
    "$PYTHON_BIN" run_inference_ov_10.py dataset_name="$DATASET" \
    >"$run_log" 2>&1 &
  local pid=$!

  # Show live output; --pid makes tail exit when inference finishes
  tail -f "$run_log" --pid="$pid" 2>/dev/null || true

  set +e; wait "$pid"; rc=$?; set -e

  if [[ $rc -ne 0 ]]; then
    log "Run FAILED (exit code $rc).  See $run_log for details."
    exit $rc
  fi

  log "Run completed successfully"
  log "Results JSON: log/sam_ov/evaluation_results/results_${DATASET}_10imgs.json"
}

main "$@"
