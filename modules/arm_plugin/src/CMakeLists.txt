# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_arm_cpu_plugin")

set(CMAKE_CXX_STANDARD 14)

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_arm_neon_optimization_flags(neon_flags)
set_source_files_properties(${SOURCES} PROPERTIES COMPILE_OPTIONS "${neon_flags}")

# adds a shared library with plugin
ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CPU"
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR arm_plugin.cpp)

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${IE_MAIN_ARM_PLUGIN_SOURCE_DIR}/include")

target_link_libraries(${TARGET_NAME} PRIVATE
    IE::inference_engine
    arm_compute::arm_compute
    arm_compute::arm_compute_core
    arm_compute::half
    ${NGRAPH_LIBRARIES}
    IE::inference_engine_transformations
    IE::inference_engine_lp_transformations
    IE::ngraph_reference
    IE::util
)

if(NOT DEFINED ENABLE_LTO)
    set(ENABLE_LTO OFF)
endif()
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})
