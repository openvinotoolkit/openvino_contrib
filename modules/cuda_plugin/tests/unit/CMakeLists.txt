# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME CudaUnitTests)

file(GLOB_RECURSE SOURCES ${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/src/*.cpp ${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE HEADERS ${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/src/*.hpp ${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/src/*.cuh)

set_source_files_properties(*.cu  PROPERTIES LANGUAGE CUDA)
set_source_files_properties(*.cuh PROPERTIES LANGUAGE CUDA)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
        CUDAPlugin
        LINK_LIBRARIES
        unitTestUtils
        GSL
        itt
        IE::inference_engine
        IE::ngraph
        IE::inference_engine_transformations
        IE::interpreter_backend
        IE::inference_engine_plugin_api
        IE::ngraph_backend
        fmt-header-only
        CUDA::cudart
        CUDA::cudnn
        CUDA::cublas
        CUDA::cutensor
        INCLUDES
        "${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/include"
        "${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/src"
        "${InferenceEngine_C_API_SOURCE_DIR}/include"
        ADD_CPPLINT
        LABELS
        CUDA
)

target_sources(${TARGET_NAME} PRIVATE ${SOURCES})

addVersionDefines(${IE_MAIN_CUDA_PLUGIN_SOURCE_DIR}/src/cuda_plugin.cpp CI_BUILD_NUMBER)

if (${CUDAToolkit_VERSION_MAJOR} LESS_EQUAL "10")
    set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 30 35 50 60 72)
else()
    set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 35 50 60 72)
endif()
