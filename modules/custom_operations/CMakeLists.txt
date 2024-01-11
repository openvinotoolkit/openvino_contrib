# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(openvino_extensions)

include(cmake/platforms.cmake)

# Find OpenVINODeveloperPackage first to compile with SDL flags
find_package(OpenVINODeveloperPackage QUIET)
if(NOT OpenVINODeveloperPackage_FOUND)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()

# set the project version using OpenVINO major, minor and patch versions
# tweak version is used to release extension independently from openvino package
if (${SKBUILD})
    set(CMAKE_PROJECT_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
    set(CMAKE_PROJECT_VERSION_TWEAK 0)
    set(CMAKE_PROJECT_VERSION "${OpenVINO_VERSION}.${CMAKE_PROJECT_VERSION_TWEAK}")
endif()

# adding CI build number
if (DEFINED ENV{CI_BUILD_NUMBER})
    set(CMAKE_PROJECT_VERSION "${CMAKE_PROJECT_VERSION}-$ENV{CI_BUILD_NUMBER}")
endif()

add_subdirectory(user_ie_extensions)
