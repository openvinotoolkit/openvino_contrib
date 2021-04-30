# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.18)

project(InferenceEngineCUDAPlugin CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(IE_MAIN_CUDA_PLUGIN_SOURCE_DIR ${InferenceEngineCUDAPlugin_SOURCE_DIR})

include(cmake/features.cmake)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

# In this example we are picking a specific tag.
# You can also pick a specific commit, if you need to.
FetchContent_Declare(GSL
                     GIT_REPOSITORY "https://github.com/microsoft/GSL"
                     GIT_TAG "v3.1.0")

FetchContent_MakeAvailable(GSL)

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
