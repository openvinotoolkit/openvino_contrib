#!/usr/bin/env bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODEL_DEFAULT="/Users/anesterov/models/yolo12/ir/yolov12n.xml"
NITER_DEFAULT=200
DEVICES_DEFAULT="CPU METAL"

print_usage() {
  cat <<EOF
Usage: $0 [--model PATH] [--niter N] [--devices "CPU METAL"]
Defaults:
  --model   ${MODEL_DEFAULT}
  --niter   ${NITER_DEFAULT}
  --devices "CPU METAL"
EOF
}

MODEL="${MODEL_DEFAULT}"
NITER="${NITER_DEFAULT}"
DEVICES="${DEVICES_DEFAULT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --niter) NITER="$2"; shift 2;;
    --devices) DEVICES="$2"; shift 2;;
    -h|--help) print_usage; exit 0;;
    *) echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

OV_BIN="/Users/anesterov/repos/openvino/bin/arm64/Release"
OV_TBB="/Users/anesterov/repos/openvino/temp/Darwin_arm64/tbb/lib"
export DYLD_LIBRARY_PATH="${OV_BIN}:${OV_TBB}:${DYLD_LIBRARY_PATH:-}"

if [[ ! -x "${OV_BIN}/benchmark_app" ]]; then
  echo "benchmark_app not found at ${OV_BIN}/benchmark_app" >&2
  exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "Model not found: ${MODEL}" >&2
  exit 1
fi

run_device() {
  local device="$1"
  local tmp
  tmp="$(mktemp)"
  trap 'rm -f "$tmp"' RETURN
  "${OV_BIN}/benchmark_app" -m "${MODEL}" -d "${device}" -api sync -niter "${NITER}" | tee "${tmp}" >/dev/null
  # Parse latency avg and throughput
  local latency throughput
  latency="$(grep -E 'Latency:' "${tmp}" | head -n1 | sed -E 's/.*avg[[:space:]]+([0-9\.]+)/\1/;t;d')"
  throughput="$(grep -E 'Throughput:' "${tmp}" | head -n1 | awk '{print $2}')"
  echo "YOLO12,DEVICE=${device},NITER=${NITER},LATENCY_MS=${latency:-NA},THROUGHPUT_FPS=${throughput:-NA}"
}

for dev in ${DEVICES}; do
  run_device "${dev}"
done
