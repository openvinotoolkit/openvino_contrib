# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME llama_cpp_e2e_tests)

include(sources.cmake)

ov_add_test_target(
    NAME ${TARGET_NAME}
    SOURCES ${LLAMA_CPP_E2E_TEST_SOURCES}
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
    CHECK_SOURCES_LISTED
    LABELS
    OV UNIT LLAMA_CPP
    )

