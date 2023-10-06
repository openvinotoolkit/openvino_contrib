# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)
project(InferenceEngineNVIDIAGpuPlugin CXX CUDA)

# Initialize CMAKE_CUDA_ARCHITECTURES when CMAKE_CUDA_COMPILER_ID is NVIDIA.
# Raise an error if CUDA_ARCHITECTURES is empty.
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
     set(CMAKE_CUDA_STANDARD 14)
endif()

if (CUDA_KERNEL_PRINT_LOG)
    add_definitions(-DCUDA_KERNEL_PRINT_LOG)
endif()
if (ENABLE_CUDNN_BACKEND_API)
    add_definitions(-DENABLE_CUDNN_BACKEND_API)
endif()

set(IE_MAIN_CUDA_PLUGIN_SOURCE_DIR ${InferenceEngineNVIDIAGpuPlugin_SOURCE_DIR})

find_package(OpenVINODeveloperPackage REQUIRED
             PATHS "${InferenceEngineDeveloperPackage_DIR}")

include(cmake/features.cmake)

set(HAS_REQUIRED REQUIRED)

if(${CMAKE_VERSION} VERSION_LESS "3.17.0")
    set(HAS_REQUIRED)
    find_package(CUDA REQUIRED)
    set(CUDAToolkit_VERSION_MAJOR ${CUDA_VERSION_MAJOR})
    set(CUDAToolkit_VERSION_MINOR ${CUDA_VERSION_MINOR})
    set(CUDAToolkit_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    foreach(LIB IN LISTS CUDA_LIBRARIES)
        if (LIB MATCHES ".*cudart.*")
            if(WIN32)
                add_library(cudart STATIC IMPORTED GLOBAL)
                # Even though we're linking against a .dll, in Windows you statically link against
                # the .lib file found under lib/x64. The .dll will be loaded at runtime automatically
                # from the PATH search.
            else()
                add_library(cudart SHARED IMPORTED GLOBAL)
            endif()
            set_target_properties(cudart PROPERTIES IMPORTED_LOCATION ${LIB})
            if(NOT WIN32)
                # list contains /usr/local/cuda-10.2/lib64/libcudart_static.a;-pthread;dl;/usr/lib/x86_64-linux-gnu/librt.so
                # and rt is not added via other ways when build contains no *.cu files
                # and dl is not added via other ways when build contains no *.cu files in some other cases
                target_link_libraries(cudart INTERFACE rt dl)
            endif()
            add_library(CUDA::cudart ALIAS cudart)
        endif()
    endforeach()
    # Search for CUDA Driver Library
    find_library(CUDA_DRIVER_PATH
                 NAMES cuda
                 HINTS "${CUDA_TOOLKIT_ROOT_DIR}" "$ENV{CUDA_PATH}"
                 PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
    if(WIN32)
        add_library(cuda_driver STATIC IMPORTED GLOBAL)
    else()
        add_library(cuda_driver SHARED IMPORTED GLOBAL)
    endif()
    set_target_properties(cuda_driver PROPERTIES IMPORTED_LOCATION "${CUDA_DRIVER_PATH}")
    add_library(CUDA::cuda_driver ALIAS cuda_driver)
    # Search for CUBLAS Library
    find_library(CUBLAS_PATH
                 NAMES cublas
                 HINTS "${CUDA_TOOLKIT_ROOT_DIR}" "$ENV{CUDA_PATH}"
                 PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
    if(WIN32)
        add_library(cublas STATIC IMPORTED GLOBAL)
    else()
        add_library(cublas SHARED IMPORTED GLOBAL)
    endif()
    set_target_properties(cublas PROPERTIES IMPORTED_LOCATION "${CUBLAS_PATH}")
    add_library(CUDA::cublas ALIAS cublas)
else()
    find_package(CUDAToolkit REQUIRED)
endif()

find_library(CUDNN_PATH
             NAMES cudnn
             ${HAS_REQUIRED}
             HINTS "$ENV{CUDNN_PATH}" "${CUDA_TOOLKIT_ROOT_DIR}" "${CUDA_PATH}" "$ENV{CUDA_PATH}"
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

if((${CUDAToolkit_VERSION_MAJOR} LESS_EQUAL 10))
    set(CUTENSOR_LIB_DIRS lib/10.2 libcutensor/10.2)
elseif((${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11) AND (${CUDAToolkit_VERSION_MINOR} LESS_EQUAL 0))
    set(CUTENSOR_LIB_DIRS lib/11.0 libcutensor/11.0)
elseif((${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11) AND (${CUDAToolkit_VERSION_MINOR} GREATER_EQUAL 1))
    set(CUTENSOR_LIB_DIRS lib/11 libcutensor/11)
else()
    set(CUTENSOR_LIB_DIRS lib/${CUDAToolkit_VERSION_MAJOR} libcutensor/${CUDAToolkit_VERSION_MAJOR})
endif()
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

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(${CUDAToolkit_VERSION_MAJOR} LESS_EQUAL 10)
        set(CMAKE_CUDA_ARCHITECTURES "30;35;50;60;72")
    elseif(${CUDAToolkit_VERSION_MAJOR} EQUAL 11 AND ${CUDAToolkit_VERSION_MINOR} LESS_EQUAL 7)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60-virtual;61;70-virtual;75;86")
    elseif(${CUDAToolkit_VERSION_MAJOR} EQUAL 11 AND ${CUDAToolkit_VERSION_MINOR} EQUAL 8)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60-virtual;61;70-virtual;75;86;89-virtual;90-virtual")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "50;60-virtual;61;70-virtual;75;86;89-virtual;90-virtual")
    endif()
endif()
message("-- [nvidia_gpu] CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")

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

    add_subdirectory(tests/unit)
    if(ENABLE_FUNCTIONAL_TESTS)
        add_subdirectory(tests/functional)
    endif()
endif()

# install

# ATTENTION: uncomment to install component
# ie_cpack(template)
