# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env bash
# run_flashocc_ov_ws.sh — Run FlashOCC E2E with pip-installed OpenVINO 2026+
#
# Setup workflow:
#   1. bash setup.sh --prepare-models --model-variant m0 --model-dir /path/to/split_f16out
#      (Generates OpenVINO IR models in split mode)
#   2. bash setup.sh --run-test --model-dir /path/to/split_f16out
#      (Creates venv, installs OpenVINO from pip, builds bev_pool, runs E2E benchmark)
#
# Or combine in one step:
#   bash setup.sh --prepare-models --run-test --model-variant m0
#
# Usage:
#   bash run_flashocc_ov_ws.sh [--num-samples N] [--ov-device GPU|CPU]
#   First run:  bash setup.sh --model-dir /path/to/split_f16out ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load paths written by setup.sh ───────────────────────────────────────────
SETUP_ENV="${SCRIPT_DIR}/setup.env"
FLASHOCC_VENV="${SCRIPT_DIR}/venv_flashocc_ws"
if [[ -f "$SETUP_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$SETUP_ENV"
fi

VENV="${FLASHOCC_VENV}"

# ── Defaults (can be overridden via CLI args) ─────────────────────────────────
MODEL_DIR="${FLASHOCC_MODEL_DIR:-${SCRIPT_DIR}/work_dirs/flashocc-r50-m0/openvino/split_f16out}"
NUM_SAMPLES=80
RUN_DURATION=0
OV_DEVICE="GPU"
OV_BEVPOOL_DEVICE="GPU"
WARMUP_FRAMES=5
OV_EXT_SO="${FLASHOCC_BEV_SO:-${SCRIPT_DIR}/openvino_extensions/bev_pool/build_ws/libopenvino_bevpool_extension.so}"
OV_GPU_CONFIG="${SCRIPT_DIR}/openvino_extensions/bev_pool/bev_pool_gpu_panterlake.xml"

usage() {
  cat <<HELP
Usage: $0 [options]
  --model-dir DIR      IR model directory with split_f16out models  (default: \$MODEL_DIR)
  --num-samples N      Frames to infer                              (default: $NUM_SAMPLES)
  --run-duration SECS  Run for at least this many seconds, cycling the batch (default: disabled)
  --ov-device DEV      OV inference device                          (default: $OV_DEVICE)
  -h|--help            Show this help
HELP
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)    MODEL_DIR="$2";    shift 2 ;;
    --num-samples)  NUM_SAMPLES="$2";  shift 2 ;;
    --run-duration) RUN_DURATION="$2"; shift 2 ;;
    --ov-device)    OV_DEVICE="$2";    shift 2 ;;
    -h|--help)      usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

# ── Validate venv ─────────────────────────────────────────────────────────────
[[ -f "${VENV}/bin/python3" ]] || {
  echo "ERROR: venv not found at ${VENV}"
  echo "  Run setup first:  bash setup.sh --model-dir /path/to/split_f16out ..."
  exit 1
}

# Locate OV libs (from venv install) and GPU plugin (from build release dir)
OV_LIBS_DIR=""
for py_lib in "${VENV}/lib/python"*; do
  [[ -d "$py_lib" ]] || continue
  candidate="${py_lib}/site-packages/openvino/libs"
  if [[ -d "$candidate" ]]; then
    OV_LIBS_DIR="$candidate"
    break
  fi
done

[[ -n "$OV_LIBS_DIR" ]] || {
  echo "ERROR: OV libs not found in venv site-packages (expected openvino/libs). Re-run setup.sh."
  exit 1
}

# Build LD_LIBRARY_PATH: venv OV libs first, then release dir (GPU plugin) if known
LD_PATH="${OV_LIBS_DIR}"
if [[ -n "${OV_RELEASE_DIR:-}" && -d "${OV_RELEASE_DIR}" ]]; then
  LD_PATH="${LD_PATH}:${OV_RELEASE_DIR}"
fi
export LD_LIBRARY_PATH="${LD_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Print run summary ─────────────────────────────────────────────────────────
echo "=== FlashOCC OpenVINO 2026.3 (gpu-flashocc-fixes) ==="
OV_VER=$("${VENV}/bin/python3" -c 'import openvino; print(openvino.__version__)' 2>/dev/null || echo "unknown")
echo "  OV version  : ${OV_VER}"
echo "  Device      : ${OV_DEVICE} / BEVpool: ${OV_BEVPOOL_DEVICE}"
echo "  Model dir   : ${MODEL_DIR}"
echo "  Frames      : ${NUM_SAMPLES}  (warmup: ${WARMUP_FRAMES})"
[[ "$RUN_DURATION" != "0" ]] && echo "  Run duration : ${RUN_DURATION}s (sample batch recycled until budget)"
echo ""

# ── Build optional arg lists ──────────────────────────────────────────────────
EXT_ARGS=()
[[ -f "${OV_EXT_SO}" ]]    && EXT_ARGS+=(--ov-extension-so    "${OV_EXT_SO}")
[[ -f "${OV_GPU_CONFIG}" ]] && EXT_ARGS+=(--ov-gpu-config-xml "${OV_GPU_CONFIG}")
[[ "$RUN_DURATION" != "0" ]] && EXT_ARGS+=(--run-duration "${RUN_DURATION}")

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"
exec "${VENV}/bin/python3" run_flashocc_ov.py \
  --model-dir         "$MODEL_DIR" \
  --num-samples       "$NUM_SAMPLES" \
  --ov-device         "$OV_DEVICE" \
  --ov-bevpool-device "$OV_BEVPOOL_DEVICE" \
  --warmup-frames     "$WARMUP_FRAMES" \
  --ov-enc-inference-precision f16 \
  --ov-trk-inference-precision f16 \
  "${EXT_ARGS[@]}" \
  "$@"
