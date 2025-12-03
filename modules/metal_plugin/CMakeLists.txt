# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(OpenVINOMetalPlugin)

find_package(OpenVINODeveloperPackage REQUIRED)

option(ENABLE_METAL_MLIR "Build MLIR-based METAL backend" ON)

if(ENABLE_METAL_MLIR)
    set(METAL_LLVM_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project")
    set(METAL_LLVM_PROJECT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project")

    # Build only what we need from llvm-project: host target + MLIR
    set(_metal_saved_build_shared_libs ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
    set(LLVM_TARGETS_TO_BUILD "Native" CACHE STRING "Targets for llvm-project" FORCE)
    set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "Enable MLIR project" FORCE)
    set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "Disable unexpected bindings" FORCE)
    set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "Skip LLVM examples" FORCE)
    set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "Skip LLVM tests" FORCE)
    set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "Skip MLIR python bindings" FORCE)
    set(LLVM_BUILD_LLVM_DYLIB OFF CACHE BOOL "Disable monolithic LLVM dylib" FORCE)
    set(LLVM_BUILD_SHARED_LIBS OFF CACHE BOOL "Build llvm-project static libs for METAL" FORCE)

    add_subdirectory(${METAL_LLVM_PROJECT_SOURCE_DIR}/llvm
                     ${METAL_LLVM_PROJECT_BINARY_DIR}/llvm
                     EXCLUDE_FROM_ALL)
    # restore parent BUILD_SHARED_LIBS state
    set(BUILD_SHARED_LIBS ${_metal_saved_build_shared_libs} CACHE BOOL "Build shared libs" FORCE)

    set(MLIR_INCLUDE_DIRS
        ${METAL_LLVM_PROJECT_SOURCE_DIR}/mlir/include
        ${METAL_LLVM_PROJECT_SOURCE_DIR}/llvm/include
        ${METAL_LLVM_PROJECT_BINARY_DIR}/llvm/include
        ${METAL_LLVM_PROJECT_BINARY_DIR}/llvm/tools/mlir/include
        CACHE INTERNAL "MLIR include directories for METAL plugin")
endif()

ov_option(ENABLE_METAL_REGISTRATION "Enables registration of METAL plugin" ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wall)
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

# install packaging (re-uses Dev Package helper)
if(OpenVINODeveloperPackage_FOUND)
    ov_cpack(metal)
endif()
