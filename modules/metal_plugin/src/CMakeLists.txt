# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_metal_plugin")

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.mm)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

# Ensure newly added per-op files are included even if glob caching misses them
list(APPEND SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/conv_kernel_ir.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/batchnorm_kernel_ir.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/mul_kernel_ir.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/msl/conv_msl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/msl/batchnorm_msl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/msl/mul_msl.cpp
)
# Exclude LayerNorm sources when the op is not available in this build.
list(FILTER SOURCES EXCLUDE REGEX "layernorm_kernel_ir.cpp")
list(FILTER SOURCES EXCLUDE REGEX "layernorm_msl.cpp")
# MLIR-only backend: exclude legacy MPSGraph-based sources
list(FILTER SOURCES EXCLUDE REGEX ".*/ops/.*")
list(FILTER SOURCES EXCLUDE REGEX ".*/graph/mps_node_context.mm")
list(FILTER SOURCES EXCLUDE REGEX ".*/graph/mps_graph_builder.mm")
list(FILTER SOURCES EXCLUDE REGEX ".*/mlir/.*")
list(FILTER HEADERS EXCLUDE REGEX ".*/mlir/.*")

if(ENABLE_METAL_MLIR)
    add_subdirectory(mlir)
endif()

# Build METAL plugin as SHARED (not MODULE) to allow normal linking/registration in OpenVINO
add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})

ov_add_version_defines(plugin/plugin.cpp ${TARGET_NAME})
target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)
ov_add_vs_version_file(NAME ${TARGET_NAME}
    FILEDESCRIPTION "OpenVINO Runtime METAL device plugin library")

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_codegen"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir"
    "${CMAKE_CURRENT_SOURCE_DIR}/msl"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops"
    "${CMAKE_CURRENT_SOURCE_DIR}/transforms"
    "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
    $<$<BOOL:${ENABLE_METAL_MLIR}>:${MLIR_INCLUDE_DIRS}>)

target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_METAL_MLIR)

# Link reference implementations for early bring-up
# TODO: replace with Metal/MPSGraph execution backend once available
target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::runtime::dev
    openvino::interpreter_backend
    openvino::reference
    $<$<BOOL:${ENABLE_METAL_MLIR}>:metal_mlir>
    "-framework MetalPerformanceShadersGraph"
    "-framework MetalPerformanceShaders"
    "-framework Metal"
    "-framework Foundation")

ov_mark_target_as_cc(${TARGET_NAME})
ov_abi_free_target(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if (ENABLE_METAL_REGISTRATION)
    # Register plugin for plugins.xml/plugins.hpp generation
    list(APPEND PLUGIN_FILES "METAL:${TARGET_NAME}")
    set(PLUGIN_FILES "${PLUGIN_FILES}" CACHE INTERNAL "" FORCE)
    set(METAL_CONFIG "" CACHE INTERNAL "" FORCE)
    set(METAL_PSEUDO_PLUGIN_FOR "" CACHE INTERNAL "" FORCE)
    set(METAL_AS_EXTENSION "" CACHE INTERNAL "" FORCE)
    add_dependencies(ov_plugins ${TARGET_NAME})
    ov_register_plugins(MAIN_TARGET ${TARGET_NAME})
endif()

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${OV_CPACK_PLUGINSDIR}
        COMPONENT metal)

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
