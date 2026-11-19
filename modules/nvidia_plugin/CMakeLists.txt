# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Same minimum as OpenVINO
cmake_minimum_required(VERSION 3.26)

# With CMP0104 NEW, project() fills CMAKE_CUDA_ARCHITECTURES with the compiler default (and checks any value set
# before it against nvcc), so only remember here whether the user chose one; the plugin default is set after project().
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(NVIDIA_GPU_DEFAULT_CUDA_ARCHITECTURES ON)
endif()

project(OpenVINONVIDIAGpuPlugin CXX CUDA)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
elseif(CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "nvidia_plugin requires CMAKE_CXX_STANDARD >= 20, but ${CMAKE_CXX_STANDARD} was set: "
        "CUDA code is always built as C++20 and must match the host code.")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# CMAKE_CUDA_STANDARD set after CUDA toolkit detection below: NVCC < 12.0 cannot compile
# C++20 device code, so a hard floor on CUDAToolkit_VERSION_MAJOR is enforced there
# instead of silently degrading to an older CUDA/C++ standard.
if(CUDA_KERNEL_PRINT_LOG)
    add_definitions(-DCUDA_KERNEL_PRINT_LOG)
endif()

if(ENABLE_CUDNN_BACKEND_API)
    add_definitions(-DENABLE_CUDNN_BACKEND_API)
endif()

find_package(OpenVINODeveloperPackage REQUIRED
    PATHS "${InferenceEngineDeveloperPackage_DIR}")

include(cmake/features.cmake)

set(HAS_REQUIRED REQUIRED)

find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_VERSION_MAJOR LESS 12)
    message(FATAL_ERROR
        "CUDAToolkit ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} was found, but "
        "nvidia_plugin requires CUDAToolkit >= 12.0: NVCC below 12.0 cannot compile C++20 device code.")
endif()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

find_library(CUDNN_PATH
             NAMES cudnn
             ${HAS_REQUIRED}
             HINTS "$ENV{CUDNN_PATH}" "${CUDA_TOOLKIT_ROOT_DIR}" "${CUDA_PATH}" "$ENV{CUDA_PATH}"
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

set(CUTENSOR_LIB_DIRS lib/${CUDAToolkit_VERSION_MAJOR} libcutensor/${CUDAToolkit_VERSION_MAJOR})

find_library(CUTENSOR_PATH
    NAMES cutensor
    ${HAS_REQUIRED}
    HINTS "$ENV{CUTENSOR_PATH}" "${CUDA_TOOLKIT_ROOT_DIR}" "${CUDA_PATH}" "$ENV{CUDA_PATH}"
    PATH_SUFFIXES nvidia/current lib64 lib/x64 lib ${CUTENSOR_LIB_DIRS})
get_filename_component(CUTENSOR_INCLUDE_DIR "${CUTENSOR_PATH}" DIRECTORY)
get_filename_component(CUTENSOR_INCLUDE_DIR "${CUTENSOR_INCLUDE_DIR}/../../include" REALPATH)

if(WIN32)
    string(REPLACE "-Zi" "-Z7" CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
    message("-- [nvidia_gpu] CMAKE_CUDA_FLAGS_DEBUG ${CMAKE_CUDA_FLAGS_DEBUG}")
endif()

message("-- [nvidia_gpu] CUTENSOR_PATH ${CUTENSOR_PATH}")
message("-- [nvidia_gpu] CUTENSOR_INCLUDE_DIR ${CUTENSOR_INCLUDE_DIR}")
message("-- [nvidia_gpu] CUDNN_PATH ${CUDNN_PATH}")
message("-- [nvidia_gpu] CUBLAS_PATH ${CUBLAS_PATH}")

add_library(CUDA::cudnn SHARED IMPORTED)
set_target_properties(CUDA::cudnn PROPERTIES IMPORTED_LOCATION "${CUDNN_PATH}")
set_target_properties(CUDA::cudnn PROPERTIES IMPORTED_IMPLIB "${CUDNN_PATH}")
add_library(CUDA::cutensor SHARED IMPORTED)
set_target_properties(CUDA::cutensor PROPERTIES IMPORTED_LOCATION "${CUTENSOR_PATH}")
set_target_properties(CUDA::cutensor PROPERTIES IMPORTED_IMPLIB "${CUTENSOR_PATH}")

if(NVIDIA_GPU_DEFAULT_CUDA_ARCHITECTURES)
    if(CUDAToolkit_VERSION_MAJOR LESS 13)
        set(NVIDIA_GPU_CUDA_ARCHITECTURES "50;60-virtual;61;70-virtual;75;86;89-virtual;90-virtual")
    else()
        # CUDA 13 dropped targets older than sm_75
        set(NVIDIA_GPU_CUDA_ARCHITECTURES "75;86;89-virtual;90-virtual")
    endif()

    # Drop the compiler default cached by project() so the plugin default stays local to this directory
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    set(CMAKE_CUDA_ARCHITECTURES "${NVIDIA_GPU_CUDA_ARCHITECTURES}")
endif()

message("-- [nvidia_gpu] CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")

include(FetchContent)

# In this example we are picking a specific tag.
# You can also pick a specific commit, if you need to.
FetchContent_Declare(GSL
    GIT_REPOSITORY "https://github.com/microsoft/GSL"
    GIT_TAG "v3.1.0")
FetchContent_MakeAvailable(GSL)
# it contains c++14 requirement, which overrides c++20 setting in CMAKE_CUDA_FLAGS
set_property(TARGET GSL PROPERTY INTERFACE_COMPILE_FEATURES)

FetchContent_Declare(fmt-header-only
    GIT_REPOSITORY "https://github.com/fmtlib/fmt"
    GIT_TAG "7.1.3")
FetchContent_MakeAvailable(fmt-header-only)
# it contains c++11 requirement, which overrides c++20 setting in CMAKE_CUDA_FLAGS
set_property(TARGET fmt-header-only PROPERTY INTERFACE_COMPILE_FEATURES)

add_subdirectory(src)
#add_subdirectory(thirdparty)

if(ENABLE_TESTS)
    include(CTest)
    enable_testing()

    add_subdirectory(tests/unit)
    if(ENABLE_FUNCTIONAL_TESTS)
        add_subdirectory(tests/functional)
    endif()
endif()

# install

# ATTENTION: uncomment to install component
# ov_cpack(template)
