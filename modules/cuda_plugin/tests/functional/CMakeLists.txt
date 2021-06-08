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
        LINK_LIBRARIES
        IE::funcSharedTests
        CUDA::cudart
        INCLUDES
        "${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/include"
        "${OpenVINO_MAIN_SOURCE_DIR}/tests"
        "${CUTENSOR_INCLUDE_DIR}"
        LABELS
        CUDA
)
target_include_directories(${TARGET_NAME} PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../include)
# [cmake:functional_tests]
