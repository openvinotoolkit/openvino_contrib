#!/bin/bash

set -exuo pipefail

TESTED_TOOL_CMD_LINE=$@

# =======================================================================================
# common
CUDASAN_KERNEL_FILTER="--filter kernel_substring=CUDAPlugin"
CUDASAN_COMMON_ARGS="--print-level=warn --print-limit=100 --error-exitcode=1"

# =======================================================================================
# memcheck
# Reports thr folowing:
#   - Errors due to out of bounds or misaligned accesses to memory by a global, local, shared or global atomic access.
#   - Errors that are reported by the hardware error reporting mechanism.
#   - Errors that occur due to incorrect use of malloc()/free() in CUDA kernels.
#   - Allocations of device memory using cudaMalloc() that have not been freed by the application.
#   - Allocations of device memory using malloc() in device code that have not been freed by the application.
CUDASAN_TOOL_ARGS="--prefix=[cuda-memcheck] --leak-check=full --check-device-heap=yes --report-api-errors=no"
echo "[cuda-memcheck] started"
time compute-sanitizer --tool=memcheck ${CUDASAN_TOOL_ARGS} ${CUDASAN_COMMON_ARGS} ${TESTED_TOOL_CMD_LINE}
echo "[cuda-memcheck] completed"

# =======================================================================================
# racecheck
#   Identify CUDA shared memory memory access race conditions.
CUDASAN_TOOL_ARGS="--prefix=[cuda-racecheck] --racecheck-report=all"
echo "[cuda-racecheck] started"
time compute-sanitizer --tool=racecheck ${CUDASAN_TOOL_ARGS} ${CUDASAN_KERNEL_FILTER} ${CUDASAN_COMMON_ARGS} ${TESTED_TOOL_CMD_LINE}
echo "[cuda-racecheck] completed"

# =======================================================================================
# synccheck
#   Whether an application is correctly using CUDA synchronization primitives, specifically
#   __syncthreads() and __syncwarp() intrinsics and their Cooperative Groups API counterparts.
CUDASAN_TOOL_ARGS="--prefix=[cuda-synccheck]"
echo "[cuda-synccheck] started"
time compute-sanitizer --tool=synccheck ${CUDASAN_TOOL_ARGS} ${CUDASAN_KERNEL_FILTER} ${CUDASAN_COMMON_ARGS} ${TESTED_TOOL_CMD_LINE}
echo "[cuda-synccheck] completed"

# =======================================================================================
# initcheck
#   Identify when device global memory is accessed without it being initialized via device side writes,
#   or via CUDA memcpy and memset API calls

#
# This check is disabled due to false positives (version 2020.3.1).
#
echo "[cuda-initcheck] disabled"
# CUDASAN_TOOL_ARGS="--prefix=[cuda-initcheck] --track-unused-memory=no"
# echo "[cuda-initcheck] started"
# time compute-sanitizer --tool=initcheck ${CUDASAN_TOOL_ARGS} ${CUDASAN_KERNEL_FILTER} ${CUDASAN_COMMON_ARGS} ${TESTED_TOOL_CMD_LINE}
# echo "[cuda-initcheck] completed"

# =======================================================================================
