# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_metal_plugin")

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.mm)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

# MLIR-only backend: exclude legacy MPSGraph-based sources
list(FILTER SOURCES EXCLUDE REGEX ".*/ops/.*")
list(FILTER SOURCES EXCLUDE REGEX ".*/graph/mps_node_context.mm")
list(FILTER SOURCES EXCLUDE REGEX ".*/graph/mps_graph_builder.mm")

if (NOT ENABLE_METAL_REGISTRATION)
    set(skip_plugin SKIP_INSTALL SKIP_REGISTRATION)
endif()

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "METAL"
              SOURCES ${SOURCES} ${HEADERS}
              ${skip_plugin}
              VERSION_DEFINES_FOR plugin/plugin.cpp
              ADD_CLANG_FORMAT)

ov_mark_target_as_cc(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_codegen"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops"
    "${CMAKE_CURRENT_SOURCE_DIR}/transforms")

target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_METAL_MLIR)

# Link reference implementations for early bring-up
# TODO: replace with Metal/MPSGraph execution backend once available
target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::runtime::dev
    openvino::interpreter_backend
    openvino::reference
    "-framework MetalPerformanceShadersGraph"
    "-framework MetalPerformanceShaders"
    "-framework Metal"
    "-framework Foundation")

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if (ENABLE_METAL_REGISTRATION)
    ov_register_plugins(MAIN_TARGET ${TARGET_NAME})
endif()

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
