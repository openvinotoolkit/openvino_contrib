# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(openvino_extensions)

include(cmake/platforms.cmake)

add_subdirectory(user_ie_extensions)
