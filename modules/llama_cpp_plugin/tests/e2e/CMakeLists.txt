# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME llama_cpp_e2e_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            llama_cpp_plugin
        LINK_LIBRARIES
            openvino::runtime::dev
            openvino::funcSharedTests
        INCLUDES
            "${OpenVINOTemplatePlugin_SOURCE_DIR}/include"
        ADD_CLANG_FORMAT
        LABELS
            OV UNIT TEMPLATE
)

