#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Build script for PointPillars OpenVINO Extension.
#
# Builds the CPU extension library:
#   ov_plugins/build/pp_extension.so
#
# GPU custom ops use SimpleGPU XML config + OpenCL kernels, no .so needed.
# The XML (pp_custom_gpu_kernels.xml) is loaded at runtime via CONFIG_FILE.
#
# Usage:
#   cd PointPillars
#   bash ov_plugins/build.sh              # auto-detect OV via Python
#   bash ov_plugins/build.sh --clean      # clean rebuild
#
# Environment variables (optional):
#   CMAKE_BUILD_TYPE : Debug, Release, RelWithDebInfo (default: Release)
#   NUM_JOBS         : parallel jobs (default: nproc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
NUM_JOBS="${NUM_JOBS:-$(nproc)}"

# --- Parse arguments ---
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

# --- Clean if requested ---
if [[ "$CLEAN" -eq 1 ]] && [[ -d "$BUILD_DIR" ]]; then
    echo "[build.sh] Cleaning ${BUILD_DIR} ..."
    rm -rf "$BUILD_DIR"
fi

# --- Auto-detect OpenVINO installation via Python ---
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

# --- Detect optional build dependencies and runtime paths ---
CMAKE_EXTRA_ARGS=""
if [[ -n "$OV_CMAKE_DIR" ]]; then
    CMAKE_EXTRA_ARGS="-DOpenVINO_DIR=${OV_CMAKE_DIR}"
fi

# The OpenCL loader usually ships with the same environment as OpenVINO, but
# without the unversioned symlink CMake looks for.
OPENCL_LIB=""
if ! OPENCL_LIB="$(python3 -c "import glob, sys; print(next(iter(sorted(glob.glob(sys.prefix + '/lib/libOpenCL.so*')) + sorted(glob.glob('/usr/lib/*/libOpenCL.so*'))), ''))" 2>/dev/null)"; then
    echo "[build.sh] WARNING: Failed to detect the OpenCL loader; continuing without -DOpenCL_LIBRARY."
elif [[ -n "$OPENCL_LIB" ]]; then
    CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DOpenCL_LIBRARY=${OPENCL_LIB}"
else
    echo "[build.sh] WARNING: OpenCL loader not found; CMake may skip the runtime helper targets."
fi

RUNTIME_RPATH=""
if ! RUNTIME_RPATH="$(python3 -c "import openvino, os, sys; print(os.path.dirname(openvino.__file__) + '/libs:' + sys.prefix + '/lib')" 2>/dev/null)"; then
    echo "[build.sh] WARNING: Failed to determine the OpenVINO runtime RPATH; continuing without -DPP_RUNTIME_RPATH."
elif [[ -n "$RUNTIME_RPATH" ]]; then
    CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DPP_RUNTIME_RPATH=${RUNTIME_RPATH}"
else
    echo "[build.sh] WARNING: OpenVINO runtime RPATH is empty; continuing without -DPP_RUNTIME_RPATH."
fi

# --- Print configuration ---
echo "============================================================"
echo " Build: PointPillars OV Extension"
echo "============================================================"
echo "  Source:     ${SCRIPT_DIR}"
echo "  Build:      ${BUILD_DIR}"
echo "  Build type: ${CMAKE_BUILD_TYPE}"
echo "  Jobs:       ${NUM_JOBS}"
echo "  OV cmake:   ${OV_CMAKE_DIR}"
echo "  OpenCL:     ${OPENCL_LIB:-not found}"
echo "  Runtime RPATH: ${RUNTIME_RPATH:-not set}"
echo "============================================================"
echo ""

# --- Generation ---
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Different constants are compiled in, and ov_export.py controls them. 
echo "[build.sh] Writing plugin constants from ov_export.py ..."
python3 "${SCRIPT_DIR}/../ov_export.py" --config-only --plugin-dir "${SCRIPT_DIR}"

# --- Configure ---
echo "[build.sh] Configuring ..."
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    ${CMAKE_EXTRA_ARGS} \
    2>&1

# --- Build ---
echo ""
echo "[build.sh] Building (${NUM_JOBS} jobs) ..."
cmake --build "${BUILD_DIR}" --parallel "${NUM_JOBS}" 2>&1

# --- Report ---
echo ""
echo "============================================================"
echo " Build complete"
echo "============================================================"

if [[ -f "${BUILD_DIR}/pp_extension.so" ]]; then
    echo "  Extension: ${BUILD_DIR}/pp_extension.so"
    ls -lh "${BUILD_DIR}/pp_extension.so"
else
    echo "  Extension: NOT FOUND (build may have failed)"
fi

echo ""
echo "Usage:"
echo "  # In Python:"
echo "  from openvino.runtime import Core"
echo "  core = Core()"
echo "  core.add_extension('${BUILD_DIR}/pp_extension.so')"
