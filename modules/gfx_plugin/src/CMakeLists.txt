# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_gfx_plugin")

include(GNUInstallDirs)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GfxSources.cmake)

function(gfx_embed_kernel_source input_file output_file symbol_name)
    file(READ "${input_file}" GFX_KERNEL_SOURCE)
    set(GFX_KERNEL_INPUT "${input_file}")
    set(GFX_KERNEL_SYMBOL "${symbol_name}")
    get_filename_component(_output_dir "${output_file}" DIRECTORY)
    file(MAKE_DIRECTORY "${_output_dir}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input_file}")
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/KernelSource.hpp.in
        "${output_file}"
        @ONLY)
endfunction()

gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/eltwise_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/eltwise_kernel_source.hpp"
    "kOpenClGeneratedEltwiseKernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/activation_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/activation_kernel_source.hpp"
    "kOpenClGeneratedActivationKernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/reduction_f32_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/reduction_f32_kernel_source.hpp"
    "kOpenClGeneratedReductionF32KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/reduction_logical_bool_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/reduction_logical_bool_kernel_source.hpp"
    "kOpenClBaselineReductionLogicalBoolKernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/softmax_f32_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/softmax_f32_kernel_source.hpp"
    "kOpenClGeneratedSoftmaxF32KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/softmax_f16_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/softmax_f16_kernel_source.hpp"
    "kOpenClGeneratedSoftmaxF16KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/softmax_f32_dynamic_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/softmax_f32_dynamic_kernel_source.hpp"
    "kOpenClGeneratedSoftmaxDynamicF32KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/softmax_f16_dynamic_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/softmax_f16_dynamic_kernel_source.hpp"
    "kOpenClGeneratedSoftmaxDynamicF16KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/pool2d_f32_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/pool2d_f32_kernel_source.hpp"
    "kOpenClGeneratedPool2DF32KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/pool2d_f16_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/pool2d_f16_kernel_source.hpp"
    "kOpenClGeneratedPool2DF16KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/interpolate_f32_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/interpolate_f32_kernel_source.hpp"
    "kOpenClGeneratedInterpolateF32KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/interpolate_f16_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/interpolate_f16_kernel_source.hpp"
    "kOpenClGeneratedInterpolateF16KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/opencl_kernels/matmul_f32_kernel.cl"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/opencl_kernels/generated/matmul_f32_kernel_source.hpp"
    "kOpenClGeneratedMatMulF32KernelSource")

gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/mpsrt_buffer_to_image_f32.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/mpsrt_buffer_to_image_f32_source.hpp"
    "kMpsrtBufferToImageF32Source")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/mpsrt_buffer_to_image_f16.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/mpsrt_buffer_to_image_f16_source.hpp"
    "kMpsrtBufferToImageF16Source")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/mpsrt_image_to_buffer_f32.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/mpsrt_image_to_buffer_f32_source.hpp"
    "kMpsrtImageToBufferF32Source")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/mpsrt_image_to_buffer_f16.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/mpsrt_image_to_buffer_f16_source.hpp"
    "kMpsrtImageToBufferF16Source")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/mpsrt_topk_pack_u32_to_i64.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/mpsrt_topk_pack_u32_to_i64_source.hpp"
    "kMpsrtTopKPackU32ToI64Source")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/mpsrt_topk_stable_i64_indices_f32.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/mpsrt_topk_stable_i64_indices_f32_source.hpp"
    "kMpsrtTopKStableI64IndicesF32Source")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/mpsrt_topk_stable_i64_indices_f16.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/mpsrt_topk_stable_i64_indices_f16_source.hpp"
    "kMpsrtTopKStableI64IndicesF16Source")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/reduction_f32_kernel.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/reduction_f32_kernel_source.hpp"
    "kMetalGeneratedReductionF32KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/reduction_logical_bool_kernel.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/reduction_logical_bool_kernel_source.hpp"
    "kMetalGeneratedReductionLogicalBoolKernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/softmax_f32_kernel.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/softmax_f32_kernel_source.hpp"
    "kMetalGeneratedSoftmaxF32KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/softmax_f16_kernel.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/softmax_f16_kernel_source.hpp"
    "kMetalGeneratedSoftmaxF16KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/logsoftmax_f32_kernel.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/logsoftmax_f32_kernel_source.hpp"
    "kMetalGeneratedLogSoftmaxF32KernelSource")
gfx_embed_kernel_source(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ir/metal_kernels/logsoftmax_f16_kernel.metal"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel_ir/metal_kernels/generated/logsoftmax_f16_kernel_source.hpp"
    "kMetalGeneratedLogSoftmaxF16KernelSource")

set(GFX_PLUGIN_COMMON_SOURCES ${GFX_PLUGIN_SOURCES})
set(GFX_PLUGIN_ENTRY_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/plugin/plugin.cpp")
list(FILTER GFX_PLUGIN_COMMON_SOURCES EXCLUDE REGEX "plugin/plugin\\.cpp$")
set(GFX_PLUGIN_METAL_SOURCES_EFFECTIVE "")
set(GFX_PLUGIN_OPENCL_SOURCES_EFFECTIVE "")
set(GFX_RUNTIME_COMMON_SOURCES ${GFX_RUNTIME_COMMON_SOURCES})

if(GFX_BACKEND_METAL_AVAILABLE)
    list(APPEND GFX_PLUGIN_METAL_SOURCES_EFFECTIVE
        ${GFX_PLUGIN_METAL_SOURCES})
    list(APPEND GFX_RUNTIME_METAL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/stage_factory.cpp)
else()
    list(APPEND GFX_PLUGIN_METAL_SOURCES_EFFECTIVE
        ${GFX_PLUGIN_METAL_STUB_SOURCES})
    list(APPEND GFX_RUNTIME_COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/stage_factory_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/op_support_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/runtime_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/profiling/profiler_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/profiling/gpu_timestamps_stub.cpp)
endif()

if(GFX_BACKEND_OPENCL_AVAILABLE)
    list(APPEND GFX_PLUGIN_OPENCL_SOURCES_EFFECTIVE
        ${GFX_PLUGIN_OPENCL_SOURCES})
    list(APPEND GFX_RUNTIME_OPENCL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/runtime/stage_factory.cpp)
else()
    list(APPEND GFX_PLUGIN_OPENCL_SOURCES_EFFECTIVE
        ${GFX_PLUGIN_OPENCL_STUB_SOURCES})
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin/gfx_backend_config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/plugin/gfx_backend_config.hpp
    @ONLY
)

if(GFX_ENABLE_METAL AND GFX_RUNTIME_METAL_SOURCES AND NOT GFX_BACKEND_METAL_AVAILABLE)
    message(FATAL_ERROR "GFX: Metal backend sources are present but Metal is not available.")
endif()

if(NOT GFX_PLUGIN_COMMON_SOURCES)
    message(FATAL_ERROR "GFX: plugin sources are missing.")
endif()

add_subdirectory(mlir)

# Build settings shared by plugin and runtime objects.
add_library(gfx_plugin_build_settings INTERFACE)
target_compile_definitions(gfx_plugin_build_settings INTERFACE
    GFX_BACKEND_METAL_AVAILABLE=$<BOOL:${GFX_BACKEND_METAL_AVAILABLE}>
    GFX_BACKEND_OPENCL_AVAILABLE=$<BOOL:${GFX_BACKEND_OPENCL_AVAILABLE}>)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(gfx_plugin_build_settings INTERFACE -Werror)
elseif(MSVC)
    target_compile_options(gfx_plugin_build_settings INTERFACE /WX)
endif()
if(ANDROID)
    target_compile_options(gfx_plugin_build_settings INTERFACE
        -ffunction-sections
        -fdata-sections
        -fvisibility=hidden
        -fvisibility-inlines-hidden)
    target_link_options(gfx_plugin_build_settings INTERFACE
        -Wl,--gc-sections
        -Wl,--icf=safe)
endif()

# Public headers are exposed to consumers via include/.
add_library(gfx_plugin_public_headers INTERFACE)
target_include_directories(gfx_plugin_public_headers INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include")

# Internal headers live under src/; generated headers are in the build tree.
add_library(gfx_plugin_private_headers INTERFACE)
target_include_directories(gfx_plugin_private_headers INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
    ${MLIR_INCLUDE_DIRS})

# Build common runtime as a separate static library (core target).
add_library(gfx_runtime_common STATIC)
target_sources(gfx_runtime_common PRIVATE
    ${GFX_RUNTIME_COMMON_HEADERS}
    ${GFX_RUNTIME_COMMON_SOURCES})
target_link_libraries(gfx_runtime_common PRIVATE
    gfx_plugin_build_settings
    gfx_plugin_public_headers
    gfx_plugin_private_headers
    openvino::runtime
    openvino::runtime::dev)
set_target_properties(gfx_runtime_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(gfx_runtime_mlir STATIC)
target_sources(gfx_runtime_mlir PRIVATE
    ${GFX_RUNTIME_MLIR_HEADERS}
    ${GFX_RUNTIME_MLIR_SOURCES}
    ${GFX_RUNTIME_METAL_MSL_HEADERS}
    ${GFX_RUNTIME_METAL_MSL_SOURCES})
target_link_libraries(gfx_runtime_mlir PRIVATE
    gfx_plugin_build_settings
    gfx_plugin_public_headers
    gfx_plugin_private_headers
    openvino::runtime
    openvino::runtime::dev)
set_target_properties(gfx_runtime_mlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(DEFINED GFX_LLVM_EXTERNAL_TARGET)
    add_dependencies(gfx_runtime_mlir ${GFX_LLVM_EXTERNAL_TARGET})
endif()

# Build common plugin sources as a separate static library (core target).
add_library(gfx_plugin_core STATIC)
target_sources(gfx_plugin_core PRIVATE
    ${GFX_PLUGIN_COMMON_SOURCES}
    ${GFX_PLUGIN_HEADERS})
target_link_libraries(gfx_plugin_core PRIVATE
    gfx_plugin_build_settings
    gfx_plugin_public_headers
    gfx_plugin_private_headers
    openvino::runtime
    openvino::runtime::dev)
target_compile_definitions(gfx_plugin_core PRIVATE IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)
set_target_properties(gfx_plugin_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Build backend plugin sources as separate static libraries.
add_library(gfx_plugin_metal STATIC)
target_sources(gfx_plugin_metal PRIVATE
    ${GFX_PLUGIN_METAL_SOURCES_EFFECTIVE}
    ${GFX_PLUGIN_METAL_HEADERS})
target_link_libraries(gfx_plugin_metal PRIVATE
    gfx_plugin_build_settings
    gfx_plugin_public_headers
    gfx_plugin_private_headers
    openvino::runtime
    openvino::runtime::dev)
set_target_properties(gfx_plugin_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(gfx_plugin_opencl STATIC)
target_sources(gfx_plugin_opencl PRIVATE
    ${GFX_PLUGIN_OPENCL_SOURCES_EFFECTIVE}
    ${GFX_PLUGIN_OPENCL_HEADERS})
target_link_libraries(gfx_plugin_opencl PRIVATE
    gfx_plugin_build_settings
    gfx_plugin_public_headers
    gfx_plugin_private_headers
    openvino::runtime
    openvino::runtime::dev)
set_target_properties(gfx_plugin_opencl PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Build GFX plugin as SHARED (not MODULE) to allow normal linking/registration in OpenVINO
add_library(${TARGET_NAME} SHARED)
target_sources(${TARGET_NAME} PRIVATE
    ${GFX_PLUGIN_HEADERS}
    ${GFX_PLUGIN_ENTRY_SOURCE})
target_link_libraries(${TARGET_NAME}
    PRIVATE gfx_plugin_build_settings
    PRIVATE gfx_plugin_private_headers
    PUBLIC gfx_plugin_public_headers)
target_link_libraries(${TARGET_NAME} PRIVATE
    gfx_plugin_core
    gfx_plugin_metal
    gfx_plugin_opencl)

# On macOS, force ".so" suffix so OpenVINO loader picks the plugin consistently.
if(APPLE)
    set_target_properties(${TARGET_NAME} PROPERTIES SUFFIX ".so")
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)
ov_add_version_defines(plugin/plugin.cpp ${TARGET_NAME})
ov_add_vs_version_file(NAME ${TARGET_NAME}
    FILEDESCRIPTION "OpenVINO Runtime GFX device plugin library")

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::runtime::dev
    gfx_mlir)

if(GFX_BACKEND_METAL_AVAILABLE)
    add_library(gfx_runtime_metal STATIC)
    target_sources(gfx_runtime_metal PRIVATE
        ${GFX_RUNTIME_COMMON_HEADERS}
        ${GFX_RUNTIME_METAL_SOURCES}
        ${GFX_RUNTIME_METAL_HEADERS})
    target_include_directories(gfx_runtime_metal PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/codegen)
    target_link_libraries(gfx_runtime_metal PRIVATE
        gfx_plugin_build_settings
        gfx_plugin_public_headers
        gfx_plugin_private_headers
        openvino::runtime
        openvino::runtime::dev)
    target_link_libraries(gfx_runtime_metal PRIVATE gfx_runtime_mlir)
    set_target_properties(gfx_runtime_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_runtime_metal)
    if(DEFINED GFX_LLVM_EXTERNAL_TARGET)
        add_dependencies(gfx_runtime_metal ${GFX_LLVM_EXTERNAL_TARGET})
    endif()
endif()

if(TARGET GFX::Metal)
    target_link_libraries(${TARGET_NAME} PRIVATE GFX::Metal)
endif()

if(GFX_BACKEND_OPENCL_AVAILABLE)
    add_library(gfx_runtime_opencl STATIC)
    target_sources(gfx_runtime_opencl PRIVATE
        ${GFX_RUNTIME_COMMON_HEADERS}
        ${GFX_RUNTIME_OPENCL_SOURCES}
        ${GFX_RUNTIME_OPENCL_HEADERS})
    target_link_libraries(gfx_runtime_opencl PRIVATE
        gfx_plugin_build_settings
        gfx_plugin_public_headers
        gfx_plugin_private_headers
        openvino::runtime
        openvino::runtime::dev)
    if(CMAKE_DL_LIBS)
        target_link_libraries(gfx_runtime_opencl PRIVATE ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(gfx_runtime_opencl PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_runtime_opencl)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    gfx_runtime_common
    gfx_runtime_mlir)

ov_mark_target_as_cc(${TARGET_NAME})
ov_abi_free_target(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(DEFINED GFX_LLVM_EXTERNAL_TARGET)
    add_dependencies(${TARGET_NAME} ${GFX_LLVM_EXTERNAL_TARGET})
endif()

if (ENABLE_GFX_REGISTRATION)
    # Register plugin for plugins.xml/plugins.hpp generation
    list(APPEND PLUGIN_FILES "GFX:${TARGET_NAME}")
    set(PLUGIN_FILES "${PLUGIN_FILES}" CACHE INTERNAL "" FORCE)
    set(GFX_CONFIG "" CACHE INTERNAL "" FORCE)
    set(GFX_PSEUDO_PLUGIN_FOR "" CACHE INTERNAL "" FORCE)
    set(GFX_AS_EXTENSION "" CACHE INTERNAL "" FORCE)
    add_dependencies(ov_plugins ${TARGET_NAME})
    ov_register_plugins(MAIN_TARGET ${TARGET_NAME})
endif()

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${OV_CPACK_PLUGINSDIR}
        COMPONENT gfx)

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/openvino/gfx_plugin
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvino
        COMPONENT gfx)
