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
SEGMENTOR_MODEL="${SEGMENTOR_MODEL:-fastsam}"
# LMO test split in Data/BOP/lmo/test/000002 currently has 200 images.
# Keep MAX_IMAGES <= 200 to evaluate the full available split.
MAX_IMAGES="${MAX_IMAGES:-10}"
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

  # Validate required BOP layout used by eval_ism_ov_bop.py
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
  log "Segmentor model: $SEGMENTOR_MODEL"
  log "Max images: $MAX_IMAGES"
  log "OV_DEVICE=$OV_DEVICE OV_SAM_DEVICE=$OV_SAM_DEVICE"
  log "Log file: $run_log"

  # Run inference in background, write output to log file directly
  # so the log is always written even if stdout is piped through head/grep.
  local rc=0
  OV_DEVICE="$OV_DEVICE" \
  OV_SAM_DEVICE="$OV_SAM_DEVICE" \
  PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n "$CONDA_ENV" \
    "$PYTHON_BIN" eval_ism_ov_bop.py \
    --bop_dir "$DATASET_DIR" \
    --ov_model_dir "$OV_MODEL_DIR" \
    --ov_device "$OV_DEVICE" \
    --segmentor_model "$SEGMENTOR_MODEL" \
    --max_images "$MAX_IMAGES" \
    --batch_size 4 \
    >"$run_log" 2>&1 &
  local pid=$!

  # Show live output; --pid makes tail exit when inference finishes
  tail -f "$run_log" --pid="$pid" 2>/dev/null || true

  set +e; wait "$pid"; rc=$?; set -e

  if [[ $rc -ne 0 ]]; then
    log "Run FAILED (exit code $rc).  See $run_log for details."
    exit $rc
  fi

  local results_dir
  results_dir="$DATASET_DIR/bop/ism_ov_${OV_DEVICE,,}_${SEGMENTOR_MODEL}_results"
  local merged_json
  merged_json="$results_dir/result_${DATASET}_${MAX_IMAGES}imgs.json"
  local merged_meta_json
  merged_meta_json="$results_dir/result_${DATASET}_${MAX_IMAGES}imgs.meta.json"
  local compat_json
  compat_json="$SCRIPT_DIR/log/sam_ov/result_${DATASET}_${MAX_IMAGES}imgs.json"
  local compat_meta_json
  compat_meta_json="$SCRIPT_DIR/log/sam_ov/result_${DATASET}_${MAX_IMAGES}imgs.meta.json"

  log "Merging ISM NPZ detections into JSON"
  RESULTS_DIR="$results_dir" \
  MERGED_JSON="$merged_json" \
  MERGED_META_JSON="$merged_meta_json" \
  COMPAT_JSON="$compat_json" \
  COMPAT_META_JSON="$compat_meta_json" \
  DATASET="$DATASET" \
  OV_DEVICE="$OV_DEVICE" \
  SEGMENTOR_MODEL="$SEGMENTOR_MODEL" \
  MAX_IMAGES="$MAX_IMAGES" \
  conda run --no-capture-output -n "$CONDA_ENV" \
  "$PYTHON_BIN" - <<'PY'
import glob
import json
import os
from model.utils import convert_npz_to_json

results_dir = os.environ["RESULTS_DIR"]
merged_json = os.environ["MERGED_JSON"]
merged_meta_json = os.environ["MERGED_META_JSON"]
compat_json = os.environ["COMPAT_JSON"]
compat_meta_json = os.environ["COMPAT_META_JSON"]
dataset = os.environ["DATASET"]
ov_device = os.environ["OV_DEVICE"]
segmentor_model = os.environ["SEGMENTOR_MODEL"]
max_images = int(os.environ["MAX_IMAGES"])

npz_paths = sorted(glob.glob(os.path.join(results_dir, "*_detection_ism.npz")))
if not npz_paths:
    raise SystemExit(f"No *_detection_ism.npz files found under {results_dir}")

all_dets = []
for i in range(len(npz_paths)):
    all_dets.extend(convert_npz_to_json(i, npz_paths))

image_keys = {
  (int(det["scene_id"]), int(det["image_id"]))
  for det in all_dets
}

meta = {
  "dataset": dataset,
  "ov_device": ov_device,
  "segmentor_model": segmentor_model,
  "max_images_requested": max_images,
  "n_detection_images": len(image_keys),
  "n_detections": len(all_dets),
}

os.makedirs(os.path.dirname(merged_json), exist_ok=True)
with open(merged_json, "w") as f:
    json.dump(all_dets, f)
with open(merged_meta_json, "w") as f:
  json.dump(meta, f, indent=2)

os.makedirs(os.path.dirname(compat_json), exist_ok=True)
with open(compat_json, "w") as f:
    json.dump(all_dets, f)
with open(compat_meta_json, "w") as f:
  json.dump(meta, f, indent=2)

print(f"[run_ism] Wrote {len(all_dets)} detections to {merged_json}")
print(f"[run_ism] Wrote metadata to {merged_meta_json}")
print(f"[run_ism] Wrote compatibility copy to {compat_json}")
print(f"[run_ism] Wrote compatibility metadata to {compat_meta_json}")
PY

  log "Run completed successfully"
  log "Results dir: $results_dir"
  log "Summary JSON: $results_dir/summary.json"
  log "Detections JSON: $merged_json"
  log "Detections metadata: $merged_meta_json"
}

main "$@"
