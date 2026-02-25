#!/usr/bin/env bash

set -e

if [ -f .env ]; then
    # Load Variables from .env if present
    source .env
fi

BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
BUILD_TYPE=${BUILD_TYPE:-Release}

ENABLE_WHEEL=${ENABLE_WHEEL:-"OFF"}
WHEEL_VERSION=${WHEEL_VERSION:-"2025.3.0"}
WHEEL_TARGET=""
if [ "$ENABLE_WHEEL" = "ON" ]; then
    WHEEL_TARGET="ie_wheel"
fi

BUILD_TARGETS=${BUILD_TARGETS:-"${WHEEL_TARGET} ov_nvidia_func_tests ov_nvidia_unit_tests openvino_nvidia_gpu_plugin benchmark_app"}
ENABLE_TESTS=${ENABLE_TESTS:-"ON"}
ENABLE_FUNCTIONAL_TESTS=${ENABLE_FUNCTIONAL_TESTS:-"ON"}

[[ -n "${OPENVINO_HOME}" ]] || { echo "OPENVINO_HOME environment variable is expected"; exit 1; }
[[ -n "${OPENVINO_CONTRIB}" ]] || { echo "OPENVINO_CONTRIB environment variable is expected"; exit 1; }
[[ -n "${OPENVINO_BUILD_PATH}" ]] || { echo "OPENVINO_BUILD_PATH environment variable is expected"; exit 1; }
[[ -d "${OPENVINO_HOME}" ]] || { echo "OPENVINO_HOME=${OPENVINO_HOME} does not exist"; exit 1; }
[[ -d "${OPENVINO_CONTRIB}" ]] || { echo "OPENVINO_CONTRIB=${OPENVINO_CONTRIB} does not exist"; exit 1; }
[[ -d "${OPENVINO_CONTRIB}/modules/nvidia_plugin" ]] || { echo "Path ${OPENVINO_CONTRIB}/modules/nvidia_plugin does not exist"; exit 1; }
[[ "$1" == "--setup" || "$1" == "--build" || "$1" == "--rebuild" ]] || {
  printf "One of the following command should be provided:\n\
  - --setup (allow configure cmake project)\n\
  - --build (allow configure cmake project and then build)\n\
  - --rebuild (allow to rebuild, clean old configure, generate new one and then build)\n";
  exit 1;
}

if [[ "$1" == "--setup" || "$1" == "--build" ]]; then
  if [[ ! -d "${OPENVINO_BUILD_PATH}" ]]; then
      mkdir -p "${OPENVINO_BUILD_PATH}"
  fi
fi
if [[ "$1" == "--rebuild" ]]; then
  rm -rf "${OPENVINO_BUILD_PATH}"
  mkdir -p "${OPENVINO_BUILD_PATH}"
fi
cd "${OPENVINO_BUILD_PATH}"
cmake "${OPENVINO_HOME}" \
      -DENABLE_NVIDIA=ON \
      -DENABLE_PLUGINS_XML=ON \
      -DENABLE_TESTS="${ENABLE_TESTS}" \
      -DENABLE_FUNCTIONAL_TESTS="${ENABLE_FUNCTIONAL_TESTS}" \
      -DENABLE_WHEEL="${ENABLE_WHEEL}" \
      -DBUILD_arm_plugin=OFF \
      -DBUILD_java_api=OFF \
      -DBUILD_llama_cpp_plugin=OFF \
      -DOPENVINO_EXTRA_MODULES="${OPENVINO_CONTRIB}/modules" \
      -DWHEEL_VERSION="${WHEEL_VERSION}" \
      -DVERBOSE_BUILD=ON \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

if [[ "$1" != "--setup" ]]; then
  cmake --build "${OPENVINO_BUILD_PATH}" --target ${BUILD_TARGETS} -j "${BUILD_JOBS}"
fi
