# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME CudaUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
        CUDAPlugin
        LINK_LIBRARIES
        unitTestUtils
        CUDAPlugin_test_static
        GSL
        ngraph_backend
        INCLUDES
        "${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/include"
        "${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/src"
        "${InferenceEngine_C_API_SOURCE_DIR}/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
        ADD_CPPLINT
        LABELS
        CUDA
)
