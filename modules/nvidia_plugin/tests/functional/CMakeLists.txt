# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_nvidia_func_tests)

# TODO: remove dependency on CPU plugin after migration to tests for API 2.0
if(CMAKE_SOURCE_DIR STREQUAL OpenVINO_SOURCE_DIR AND ENABLE_INTEL_CPU)
    set(OPENVINO_ADDITIONAL_DEPENDENCIES openvino_intel_cpu_plugin)
endif()

ov_add_test_target(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            openvino_nvidia_gpu_plugin
            ${OPENVINO_ADDITIONAL_DEPENDENCIES}
        LINK_LIBRARIES_WHOLE_ARCHIVE
            openvino_nvidia_gpu_plugin_obj
        LINK_LIBRARIES
            openvino::funcSharedTests
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ADD_CLANG_FORMAT
        LABELS
           CUDA
)

if(ENABLE_PROXY)
    target_compile_definitions(${TARGET_NAME} PUBLIC PROXY_PLUGIN_ENABLED)
endif()
