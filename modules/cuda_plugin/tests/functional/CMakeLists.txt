# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:functional_tests]
set(TARGET_NAME CudaFuncTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
        openvino_cuda_plugin
        LINK_LIBRARIES_WHOLE_ARCHIVE
        openvino_cuda_plugin_obj
        LINK_LIBRARIES
        IE::funcSharedTests
        gmock
        INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        LABELS
        CUDA
)
# [cmake:functional_tests]
