# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_metal_plugin")

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.mm)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

# Ensure newly added per-op files are included even if glob caching misses them
list(APPEND SOURCES
    # Legacy per-kernel compiler helpers are not used by the object-based MLIR path.
    # Handwritten MSL removed; MLIR->MSL used instead
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_dtype.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_conv.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_conv3d.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_group_conv.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_activations.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_elementwise.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_pooling.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_softmax.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_split.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_matmul.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_concat.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_reshape.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_batchnorm.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_convert.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_interpolate.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_depth_to_space.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_slice.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_gather.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_gather_elements.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_gathernd.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_scatter_elements_update.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_scatter_nd_update.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_shapeof.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_pad.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_tile.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_broadcast.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_range.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_reverse.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_space_to_depth.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_topk.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_factory.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin/remote_stub.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/profiling/metal_profiler.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/profiling/metal_gpu_timestamps.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/profiling/metal_profiling_report.cpp
)
list(APPEND HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_conv.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_conv3d.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_group_conv.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_activations.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_elementwise.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_pooling.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_softmax.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_split.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_matmul.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_concat.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_reshape.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_batchnorm.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_convert.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_interpolate.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_depth_to_space.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_slice.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_gather.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_gather_elements.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_gathernd.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_scatter_elements_update.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_scatter_nd_update.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_shapeof.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_pad.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_tile.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_broadcast.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_range.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_reverse.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_space_to_depth.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_topk.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_op_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/profiling/metal_profiler.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/profiling/metal_profiler_config.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/profiling/metal_gpu_timestamps.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/profiling/metal_profiling_report.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin/metal_properties.hpp
)
# Remove legacy MLIR dispatcher (per-op codegen is used instead).
list(FILTER SOURCES EXCLUDE REGEX "codegen_dispatcher.cpp")
# Remove legacy reshape codegen that depends on KernelOp.
list(FILTER SOURCES EXCLUDE REGEX "reshape_codegen.cpp")
# Remove legacy Kernel IR builders and handwritten kernel compilers.
list(FILTER SOURCES EXCLUDE REGEX "kernel_ir/.*\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "kernel_codegen/metal_kernel_compiler_.*\\.mm$")
# Remove legacy MLIR builders that depend on KernelOp.
list(FILTER SOURCES EXCLUDE REGEX "mlir_builder_concat\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "mlir_builder_split\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "mlir_builder_convert\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "mlir_builder_interpolate\\.cpp$")
# Keep the common Metal compiler helper.
list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kernel_codegen/metal_kernel_compiler_common.mm)
# MLIR-only backend: source lists already limited to existing files; legacy ops/graph were removed.

if(ENABLE_METAL_MLIR)
    add_subdirectory(mlir)
    add_subdirectory(mlir_codegen)
endif()

# Build METAL plugin as SHARED (not MODULE) to allow normal linking/registration in OpenVINO
add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})

# On macOS, force ".so" suffix so OpenVINO loader picks the plugin consistently.
if(APPLE)
    set_target_properties(${TARGET_NAME} PROPERTIES SUFFIX ".so")
endif()

ov_add_version_defines(plugin/plugin.cpp ${TARGET_NAME})
target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)
ov_add_vs_version_file(NAME ${TARGET_NAME}
    FILEDESCRIPTION "OpenVINO Runtime METAL device plugin library")

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_codegen"
    "${CMAKE_CURRENT_SOURCE_DIR}/msl"
    "${CMAKE_CURRENT_SOURCE_DIR}/transforms"
    "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
    $<$<BOOL:${ENABLE_METAL_MLIR}>:${MLIR_INCLUDE_DIRS}>)

target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_METAL_MLIR)

# Link reference implementations for early bring-up
target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::runtime::dev
    openvino::interpreter_backend
    openvino::reference
    $<$<BOOL:${ENABLE_METAL_MLIR}>:metal_mlir>
    $<$<BOOL:${ENABLE_METAL_MLIR}>:metal_mlir_codegen>
    "-framework Metal"
    "-framework Foundation")

ov_mark_target_as_cc(${TARGET_NAME})
ov_abi_free_target(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(DEFINED METAL_LLVM_EXTERNAL_TARGET)
    add_dependencies(${TARGET_NAME} ${METAL_LLVM_EXTERNAL_TARGET})
endif()

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
