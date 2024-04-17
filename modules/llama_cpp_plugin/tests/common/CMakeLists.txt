# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

project(llama_cpp_test_common)

add_library(llama_cpp_test_common STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llm_inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarking.cpp
    )
target_include_directories(llama_cpp_test_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llama_cpp_test_common gtest common_test_utils)
