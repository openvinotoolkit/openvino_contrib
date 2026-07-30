#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Build script for CDPN OpenVINO Extension.
#
# Builds the CPU extension library:
#   ov_plugins/build/cdpn_extension.so
#
# GPU custom ops use SimpleGPU XML config + OpenCL kernels - no .so needed.
# The XML (cdpn_custom_gpu_kernels.xml) is loaded at runtime via CONFIG_FILE.
#
# Usage:
#   cd CDPN_ICCV2019_ZhigangLi
#   bash ov_plugins/build.sh              # auto-detect OV via Python
#   bash ov_plugins/build.sh --clean      # clean rebuild
#
# Environment variables (optional):
#   CMAKE_BUILD_TYPE - Debug, Release, RelWithDebInfo (default: Release)
#   NUM_JOBS        - parallel jobs (default: nproc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
NUM_JOBS="${NUM_JOBS:-$(nproc)}"

# Parse arguments
CLEAN=0

for arg in "$@"; do
    case "$arg" in
        --clean)
            CLEAN=1
            ;;
        --help|-h)
            echo "Usage: $0 [--clean]"
            echo ""
            echo "Options:"
            echo "  --clean           Remove build directory before building"
            echo ""
            echo "OpenVINO is auto-detected via 'python3 -c \"import openvino\"'."
            echo ""
            echo "Environment variables:"
            echo "  CMAKE_BUILD_TYPE  Build type (default: Release)"
            echo "  NUM_JOBS          Parallel jobs (default: nproc)"
            exit 0
            ;;
    esac
done

# Clean if requested
if [[ "$CLEAN" -eq 1 ]] && [[ -d "$BUILD_DIR" ]]; then
    echo "[build.sh] Cleaning ${BUILD_DIR} ..."
    rm -rf "$BUILD_DIR"
fi

# Auto-detect OpenVINO installation via Python
find_ov_cmake_dir() {
    python3 -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'cmake'))" 2>/dev/null
}

OV_CMAKE_DIR="$(find_ov_cmake_dir || true)"
if [[ -n "$OV_CMAKE_DIR" ]] && [[ -f "${OV_CMAKE_DIR}/OpenVINOConfig.cmake" ]]; then
    : # found
else
    echo "[build.sh] ERROR: Cannot find OpenVINOConfig.cmake"
    echo "[build.sh] Ensure 'python3 -c \"import openvino\"' works in this environment."
    exit 1
fi

# Print configuration
echo "============================================================"
echo " CDPN OV Extension - Build"
echo "============================================================"
echo "  Source:     ${SCRIPT_DIR}"
echo "  Build:      ${BUILD_DIR}"
echo "  Build type: ${CMAKE_BUILD_TYPE}"
echo "  Jobs:       ${NUM_JOBS}"

CMAKE_EXTRA_ARGS=""
if [[ -n "$OV_CMAKE_DIR" ]]; then
    echo "  OV cmake:   ${OV_CMAKE_DIR}"
    CMAKE_EXTRA_ARGS="-DOpenVINO_DIR=${OV_CMAKE_DIR}"
else
    echo "  OV cmake:   (auto-detect via CMAKE_PREFIX_PATH)"
fi
echo "============================================================"
echo ""

# Configure
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[build.sh] Configuring ..."
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    ${CMAKE_EXTRA_ARGS} \
    2>&1

# Build
echo ""
echo "[build.sh] Building (${NUM_JOBS} jobs) ..."
cmake --build "${BUILD_DIR}" --parallel "${NUM_JOBS}" 2>&1

# Report
echo ""
echo "============================================================"
echo " Build complete"
echo "============================================================"

if [[ -f "${BUILD_DIR}/cdpn_extension.so" ]]; then
    echo "  Extension: ${BUILD_DIR}/cdpn_extension.so"
    ls -lh "${BUILD_DIR}/cdpn_extension.so"
else
    echo "  Extension: NOT FOUND (build may have failed)"
fi

echo ""
echo "Usage:"
echo "  # In Python:"
echo "  import openvino as ov"
echo "  core = ov.Core()"
echo "  core.add_extension('${BUILD_DIR}/cdpn_extension.so')"
echo ""
