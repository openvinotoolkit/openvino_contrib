# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_nvidia_unit_tests)

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
ov_add_test_target(
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
            openvino::gtest
            openvino::gtest_main
            openvino::gmock
            openvino::commonTestUtils
        ADD_CPPLINT
        ADD_CLANG_FORMAT
        LABELS
            CUDA
)
set_source_files_properties(*.cu *.cuh PROPERTIES LANGUAGE CUDA)

set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# unitTestUtils brings in USE_STATIC_IE element in INTERFACE_COMPILE_DEFINITIONS, which breaks
# INFERENCE_ENGINE_API_CLASS() and friends since we really link to ie dlls.
# there's no easy way to override it(something like "remove from definitions" property on client)
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE -UUSE_STATIC_IE)
else(NOT CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE /U USE_STATIC_IE)
endif(NOT CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
