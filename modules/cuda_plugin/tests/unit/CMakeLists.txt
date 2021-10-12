# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME CudaUnitTests)

set(EXCLUDED_SOURCE_PATHS)
if(${CUDAToolkit_VERSION_MAJOR} EQUAL 10)
	set(EXCLUDED_SOURCE_PATHS
		EXCLUDED_SOURCE_PATHS
		concat.cpp # uses concat.cu 
		operation_registry.cpp # uses saxpy.cu, could use something else
	)
endif(${CUDAToolkit_VERSION_MAJOR} EQUAL 10)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES_WHOLE_ARCHIVE
        CUDAPlugin_OBJ
        ${EXCLUDED_SOURCE_PATHS}
        LINK_LIBRARIES
        IE::funcSharedTests
        gmock
        ngraphFunctions
        ADD_CPPLINT
        LABELS
        CUDA
)
# unitTestUtils brings in USE_STATIC_IE element in INTERFACE_COMPILE_DEFINITIONS, which breaks
# INFERENCE_ENGINE_API_CLASS() and friends since we really link to ie dlls.
# there's no easy way to override it(something like "remove from definitions" property on client)
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE -UUSE_STATIC_IE)
else(NOT CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE /U USE_STATIC_IE)
endif(NOT CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
