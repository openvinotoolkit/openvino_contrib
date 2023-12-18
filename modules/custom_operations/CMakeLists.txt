# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

set(PROJECT_VERSION 2023.3.0) # aligned with OpenVINO version
set(PROJECT_VERSION_PATCH 0)

project(openvino_extensions VERSION "${PROJECT_VERSION}.${PROJECT_VERSION_PATCH}")

include(cmake/platforms.cmake)

# Find OpenVINODeveloperPackage first to compile with SDL flags
find_package(OpenVINODeveloperPackage QUIET)
if(NOT OpenVINODeveloperPackage_FOUND)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()
set(CMAKE_PROJECT_VERSION "${OpenVINO_VERSION}.${PROJECT_VERSION_PATCH}")

message("${CMAKE_PROJECT_NAME} version is ${CMAKE_PROJECT_VERSION}")

add_subdirectory(user_ie_extensions)
