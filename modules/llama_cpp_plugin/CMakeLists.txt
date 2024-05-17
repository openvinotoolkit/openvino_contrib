# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(LlamaCppPlugin)

find_package(OpenVINODeveloperPackage REQUIRED)

ov_option(ENABLE_LLAMA_CPP_PLUGIN_REGISTRATION "Enables registration of LLAMA_CPP plugin" OFF)

add_subdirectory(src)

include(FetchContent)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp
    GIT_TAG        b2417
    )

FetchContent_MakeAvailable(llama_cpp)

if(ENABLE_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests/common)
    add_subdirectory(tests/e2e)
    add_subdirectory(tests/functional)
endif()

# install

if(OpenVINODeveloperPackage_FOUND)
    ov_cpack(LlamaCppPlugin)
endif()
