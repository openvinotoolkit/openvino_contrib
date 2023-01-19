# Copyright (C) 2019-2023 Intel Corporation
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

file(GLOB_RECURSE
	 CUDA_SOURCES
	 RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
	 CONFIGURE_DEPENDS
	 *.cu
)
addIeTargetTest(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        OBJECT_FILES
            ${CUDA_SOURCES}
        LINK_LIBRARIES_WHOLE_ARCHIVE
            openvino_nvidia_gpu_plugin_obj
        EXCLUDED_SOURCE_PATHS
            ${EXCLUDED_SOURCE_PATHS}
        LINK_LIBRARIES
            gtest
            gtest_main
            gmock
            ngraphFunctions
            commonTestUtils_s
        ADD_CPPLINT
        LABELS
            CUDA
)
set_source_files_properties(*.cu *.cuh PROPERTIES LANGUAGE CUDA)
if (${CUDAToolkit_VERSION_MAJOR} LESS_EQUAL "10")
	set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 30 35 50 60 72)
else()
	set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 35 50 60 72)
endif()
# unitTestUtils brings in USE_STATIC_IE element in INTERFACE_COMPILE_DEFINITIONS, which breaks
# INFERENCE_ENGINE_API_CLASS() and friends since we really link to ie dlls.
# there's no easy way to override it(something like "remove from definitions" property on client)
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE -UUSE_STATIC_IE)
else(NOT CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE /U USE_STATIC_IE)
endif(NOT CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
