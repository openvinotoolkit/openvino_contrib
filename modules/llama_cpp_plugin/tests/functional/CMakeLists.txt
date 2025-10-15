# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME llama_cpp_func_tests)

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDENCIES
    llama_cpp_plugin
    LINK_LIBRARIES
    openvino::runtime::dev
    common_test_utils
    gtest
    llama_cpp_test_common
    INCLUDES
    "${LlamaCppPlugin_SOURCE_DIR}/include"
    "${LlamaCppPlugin_SOURCE_DIR}/tests/common/include"
    ADD_CLANG_FORMAT
    LABELS
    OV UNIT LLAMA_CPP
    )

