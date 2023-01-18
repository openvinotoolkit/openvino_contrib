# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:functional_tests]
set(TARGET_NAME CudaFuncTests)

if (ENABLE_INTEL_CPU)
    set(OPENVINO_ADDITIONAL_DEPENDENCIES openvino_intel_cpu_plugin)
else(ENABLE_INTEL_CPU)
    set(OPENVINO_ADDITIONAL_DEPENDENCIES)
endif (ENABLE_INTEL_CPU)

addIeTargetTest(
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
            gmock
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
        LABELS
           CUDA
)
# [cmake:functional_tests]
