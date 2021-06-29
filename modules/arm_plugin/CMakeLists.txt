# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(InferenceEngineArmPlugin)

set(IE_MAIN_ARM_PLUGIN_SOURCE_DIR ${InferenceEngineArmPlugin_SOURCE_DIR})

find_package(InferenceEngineDeveloperPackage REQUIRED)

if(NOT ARM AND NOT AARCH64)
    message(WARNING "ARM CPU plugin can be built only for ARM or ARM64 target. Exiting..")
    return()
endif()

include(cmake/features.cmake)

add_subdirectory(thirdparty)
add_subdirectory(src)

if(ENABLE_TESTS)
    include(CTest)
    enable_testing()

    disable_deprecated_warnings()

    if(ENABLE_FUNCTIONAL_TESTS)
        add_subdirectory(tests/functional)
    endif()
endif()

# install

ie_cpack(cpu)
