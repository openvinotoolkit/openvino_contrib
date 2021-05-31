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
# it looks like cmake 3.13 can't derive cudnn dep from elsewhere
        CUDA::cudnn
        GSL
        fmt-header-only
        ngraph_backend
        CUDA::cublas
        INCLUDES
        "${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/include"
        "${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/src"
        "${InferenceEngine_C_API_SOURCE_DIR}/include"
        ADD_CPPLINT
        LABELS
        CUDA
)
if (${CUDAToolkit_VERSION_MAJOR} LESS_EQUAL "10")
    set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 30 35 50 60 72)
else()
    set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 35 50 60 72)
endif()