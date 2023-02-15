# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_arm_cpu_plugin")

add_subdirectory(transformations)
add_subdirectory(opset)

file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/arm_converter/*.cpp)
file(GLOB SOURCES_CORE
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(APPEND SOURCES ${SOURCES_CORE})

file(GLOB_RECURSE HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/arm_converter/*.hpp)
file(GLOB HEADERS_CORE
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
list(APPEND HEADERS ${HEADERS_CORE})

ie_arm_neon_optimization_flags(neon_flags)
set_source_files_properties(${SOURCES} PROPERTIES COMPILE_OPTIONS "${neon_flags}")

if(TARGET openvino_intel_cpu_plugin)
    set(device_name "ARM_CPU")
else()
    set(device_name "CPU")
endif()

# adds a shared library with plugin
ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "${device_name}"
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR arm_plugin.cpp)

target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${IE_MAIN_ARM_PLUGIN_SOURCE_DIR}/include")

target_include_directories(${TARGET_NAME} INTERFACE
        "${IE_MAIN_ARM_PLUGIN_SOURCE_DIR}/thirdparty/external_kernels/ncnn")
target_link_libraries(${TARGET_NAME} PRIVATE
        openvino_arm_cpu_external_kernels
        openvino_arm_cpu_transformations
        openvino_arm_cpu_opset
        arm_compute::arm_compute
        arm_compute::half
        openvino::ngraph_reference
        openvino::util)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})
