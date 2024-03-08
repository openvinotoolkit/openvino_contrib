#!/bin/bash

set -e
# What we want to do is build the llama.cpp dependency for different backends and have a separate plugin for each such build type.
# Sadly, CMake does not reliably allow to add_subdirectory multiple times in the same build tree, let alone with different options,
# since this would lead to "duplicate targets". There doesn't seem to be a solution to this problem even still. Thus, will have to
# invoke the cmake configure and build stage separately for each llama.cpp backend type.

BUILD_TYPE=$1
COMMON_OPTS="-DOpenVINODeveloperPackage_DIR=/home/vshampor/work/openvino/build -DCMAKE_EXPORT_COMPILE_COMMANDS=1"

# Regular CPU build of llama.cpp
cmake -S ./ -B ./build/cpu/ ${COMMON_OPTS} "$@"
cmake --build ./build/cpu/ -j --target llama --target llama_cpp_plugin


# CUDA build
cmake -S ./ -B ./build/cuda/ -DLLAMA_CUBLAS=1 -DPLUGIN_DEVICE_NAME="LLAMA_CPP_CUDA" -DPLUGIN_LIBRARY_NAME="llama_cpp_cuda_plugin" -DLLAMA_TARGET_NAME="llama_cuda" ${COMMON_OPTS} "$@"
cmake --build ./build/cuda/ -j --target llama_cuda --target llama_cpp_cuda_plugin
