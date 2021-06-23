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
        CUDAPlugin
        LINK_LIBRARIES_WHOLE_ARCHIVE
        CUDAPlugin_OBJ
        LINK_LIBRARIES
        IE::funcSharedTests
        INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        LABELS
        CUDA
)
# [cmake:functional_tests]
