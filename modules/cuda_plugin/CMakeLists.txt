# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)
if(CMAKE_MAJOR_VERSION EQUAL 3 AND CMAKE_MINOR_VERSION EQUAL 17)
    message(FATAL_ERROR "cmake versions 3.17 is not supported, please either upgrade or downgrade")
endif(CMAKE_MAJOR_VERSION EQUAL 3 AND CMAKE_MINOR_VERSION EQUAL 17)
project(InferenceEngineCUDAPlugin CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# cmake 3.13 doesn't understand CMAKE_CUDA_STANDARD 17 yet
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CUDA_FLAGS "-std=c++17")
endif(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

set(IE_MAIN_CUDA_PLUGIN_SOURCE_DIR ${InferenceEngineCUDAPlugin_SOURCE_DIR})

include(cmake/features.cmake)

set(HAS_REQUIRED REQUIRED)

if(${CMAKE_VERSION} VERSION_LESS "3.17.0")
    set(HAS_REQUIRED)
    find_package(CUDA REQUIRED)
    set(CUDAToolkit_VERSION_MAJOR ${CUDA_VERSION_MAJOR})
    set(CUDAToolkit_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    foreach(LIB IN LISTS CUDA_LIBRARIES)
        if (LIB MATCHES ".*cudart.*")
            add_library(cudart SHARED IMPORTED GLOBAL)
            set_target_properties(cudart PROPERTIES IMPORTED_LOCATION ${LIB})
            add_library(CUDA::cudart ALIAS cudart)
        endif()
    endforeach()
    # Search for CUDA Driver Library
    find_library(CUDA_DRIVER_PATH
                 NAMES cuda
                 HINTS "${CUDA_TOOLKIT_ROOT_DIR}" "$ENV{CUDA_PATH}"
                 PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
    add_library(cuda_driver SHARED IMPORTED GLOBAL)
    set_target_properties(cuda_driver PROPERTIES IMPORTED_LOCATION "${CUDA_DRIVER_PATH}")
    add_library(CUDA::cuda_driver ALIAS cuda_driver)
    # Search for CUBLAS Library
    find_library(CUBLAS_PATH
                 NAMES cublas
                 HINTS "${CUDA_TOOLKIT_ROOT_DIR}" "$ENV{CUDA_PATH}"
                 PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
    add_library(cublas SHARED IMPORTED GLOBAL)
    set_target_properties(cublas PROPERTIES IMPORTED_LOCATION "${CUBLAS_PATH}")
    add_library(CUDA::cublas ALIAS cublas)
else()
    find_package(CUDAToolkit REQUIRED)
endif()

find_library(CUDNN_PATH
             NAMES cudnn
             ${HAS_REQUIRED}
             HINTS "$ENV{CUDNN_PATH}"  "${CUDA_TOOLKIT_ROOT_DIR}" "${CUDA_PATH}" "$ENV{CUDA_PATH}"
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

find_library(CUTENSOR_PATH
             NAMES cutensor
             ${HAS_REQUIRED}
             HINTS "$ENV{CUTENSOR_PATH}" "${CUDA_TOOLKIT_ROOT_DIR}" "${CUDA_PATH}" "$ENV{CUDA_PATH}" 
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib lib/11)
get_filename_component(CUTENSOR_INCLUDE_DIR "${CUTENSOR_PATH}" DIRECTORY)
get_filename_component(CUTENSOR_INCLUDE_DIR "${CUTENSOR_INCLUDE_DIR}/../../include" REALPATH)

message("-- [CUDAPlugin] CUTENSOR_PATH ${CUTENSOR_PATH}")
message("-- [CUDAPlugin] CUTENSOR_INCLUDE_DIR ${CUTENSOR_INCLUDE_DIR}")
message("-- [CUDAPlugin] CUDNN_PATH ${CUDNN_PATH}")
message("-- [CUDAPlugin] CUBLAS_PATH ${CUBLAS_PATH}")

add_library(CUDA::cudnn SHARED IMPORTED)
set_target_properties(CUDA::cudnn PROPERTIES IMPORTED_LOCATION "${CUDNN_PATH}")
set_target_properties(CUDA::cudnn PROPERTIES IMPORTED_IMPLIB "${CUDNN_PATH}")
add_library(CUDA::cutensor SHARED IMPORTED)
set_target_properties(CUDA::cutensor PROPERTIES IMPORTED_LOCATION "${CUTENSOR_PATH}")
set_target_properties(CUDA::cutensor PROPERTIES IMPORTED_IMPLIB "${CUTENSOR_PATH}")

include(FetchContent)

# In this example we are picking a specific tag.
# You can also pick a specific commit, if you need to.
FetchContent_Declare(GSL
                     GIT_REPOSITORY "https://github.com/microsoft/GSL"
                     GIT_TAG "v3.1.0")
if(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    FetchContent_GetProperties(GSL)
    if(NOT GSL_POPULATED)
        FetchContent_Populate(GSL)
        message(STATUS "gsl_SOURCE_DIR is ${gsl_SOURCE_DIR}")
        add_subdirectory(${gsl_SOURCE_DIR} ${gsl_BINARY_DIR})
    endif()
else()
    FetchContent_MakeAvailable(GSL)
endif()
# it contains c++14 requirement, which overrides c++17 setting in CMAKE_CUDA_FLAGS
set_property(TARGET GSL PROPERTY INTERFACE_COMPILE_FEATURES)

FetchContent_Declare(fmt-header-only
                     GIT_REPOSITORY "https://github.com/fmtlib/fmt"
                     GIT_TAG "7.1.3")
if(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    FetchContent_GetProperties(fmt-header-only)
    if(NOT fmt-header-only_POPULATED)
        FetchContent_Populate(fmt-header-only)
        message(STATUS "fmt-header-only_SOURCE_DIR is ${fmt-header-only_SOURCE_DIR}")
        add_subdirectory(${fmt-header-only_SOURCE_DIR} ${fmt-header-only_BINARY_DIR})
    endif()
else()
    FetchContent_MakeAvailable(fmt-header-only)
endif()
# it contains c++11 requirement, which overrides c++17 setting in CMAKE_CUDA_FLAGS
set_property(TARGET fmt-header-only PROPERTY INTERFACE_COMPILE_FEATURES)

add_subdirectory(src)
#add_subdirectory(thirdparty)

if(ENABLE_TESTS)
    include(CTest)
    enable_testing()

    disable_deprecated_warnings()

    add_subdirectory(tests/unit)
    if(ENABLE_FUNCTIONAL_TESTS)
        add_subdirectory(tests/functional)
    endif()
endif()

# install

# ATTENTION: uncomment to install component
# ie_cpack(template)
