# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_gfx_plugin")

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GfxSources.cmake)

set(GFX_PLUGIN_SOURCES ${GFX_PLUGIN_SOURCES})
set(GFX_RUNTIME_COMMON_SOURCES ${GFX_RUNTIME_COMMON_SOURCES})

if(GFX_BACKEND_METAL_AVAILABLE)
    list(APPEND GFX_PLUGIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/plugin/infer_request.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/plugin/infer_io_metal.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/plugin/compiled_model_backend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/plugin/remote_context.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/plugin/device_info.mm)
    list(APPEND GFX_RUNTIME_METAL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/stage_factory.cpp)
else()
    list(APPEND GFX_PLUGIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/plugin/infer_request_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/plugin/compiled_model_backend_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/plugin/device_info_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/plugin/remote_context_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/plugin/remote_tensor_stub.cpp)
    list(APPEND GFX_RUNTIME_COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/stage_factory_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/op_support_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/runtime_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/profiling/profiler_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/runtime/profiling/gpu_timestamps_stub.cpp)
endif()

if(GFX_BACKEND_VULKAN_AVAILABLE)
    list(APPEND GFX_PLUGIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/plugin/infer_request.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/plugin/infer_io_vulkan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/plugin/compiled_model_backend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/plugin/remote_tensor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/plugin/remote_context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/plugin/device_info.cpp)
    list(APPEND GFX_RUNTIME_VULKAN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/runtime/stage_factory.cpp)
else()
    list(APPEND GFX_PLUGIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/plugin/infer_request_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/plugin/compiled_model_backend_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/plugin/device_info_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/plugin/remote_context_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/plugin/remote_tensor_stub.cpp)
    list(APPEND GFX_RUNTIME_COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/runtime/stage_factory_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/runtime/op_support_stub.cpp)
endif()

if(GFX_BACKEND_METAL_AVAILABLE)
    list(APPEND GFX_PLUGIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/plugin/remote_tensor.mm)
endif()

if(NOT GFX_BACKEND_VULKAN_AVAILABLE)
    list(APPEND GFX_RUNTIME_COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/runtime/memory_api_stub.cpp)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin/gfx_backend_config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/plugin/gfx_backend_config.hpp
    @ONLY
)

if(GFX_ENABLE_METAL AND GFX_RUNTIME_METAL_SOURCES AND NOT GFX_BACKEND_METAL_AVAILABLE)
    message(FATAL_ERROR "GFX: Metal backend sources are present but Metal is not available.")
endif()

if(NOT GFX_PLUGIN_SOURCES)
    message(FATAL_ERROR "GFX: plugin sources are missing.")
endif()

if(ENABLE_GFX_MLIR)
    add_subdirectory(mlir)
    add_subdirectory(mlir_codegen)
endif()

# Build settings shared by plugin and runtime objects.
add_library(gfx_plugin_build_settings INTERFACE)
target_include_directories(gfx_plugin_build_settings INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends"
    "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
    "${CMAKE_CURRENT_SOURCE_DIR}/mlir_codegen"
    "${CMAKE_CURRENT_SOURCE_DIR}/transforms"
    $<$<BOOL:${ENABLE_GFX_MLIR}>:${MLIR_INCLUDE_DIRS}>)
target_compile_definitions(gfx_plugin_build_settings INTERFACE
    ENABLE_GFX_MLIR=$<BOOL:${ENABLE_GFX_MLIR}>
    GFX_BACKEND_METAL_AVAILABLE=$<BOOL:${GFX_BACKEND_METAL_AVAILABLE}>
    GFX_BACKEND_VULKAN_AVAILABLE=$<BOOL:${GFX_BACKEND_VULKAN_AVAILABLE}>)

# Build GFX plugin as SHARED (not MODULE) to allow normal linking/registration in OpenVINO
add_library(${TARGET_NAME} SHARED)
target_sources(${TARGET_NAME} PRIVATE ${GFX_PLUGIN_SOURCES} ${GFX_PLUGIN_HEADERS} ${GFX_RUNTIME_COMMON_SOURCES})
target_link_libraries(${TARGET_NAME} PRIVATE gfx_plugin_build_settings)

# On macOS, force ".so" suffix so OpenVINO loader picks the plugin consistently.
if(APPLE)
    set_target_properties(${TARGET_NAME} PROPERTIES SUFFIX ".so")
endif()

ov_add_version_defines(plugin/plugin.cpp ${TARGET_NAME})
target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)
ov_add_vs_version_file(NAME ${TARGET_NAME}
    FILEDESCRIPTION "OpenVINO Runtime GFX device plugin library")

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::runtime::dev
    $<$<BOOL:${ENABLE_GFX_MLIR}>:gfx_mlir>
    $<$<BOOL:${ENABLE_GFX_MLIR}>:gfx_mlir_codegen>)

if(GFX_BACKEND_METAL_AVAILABLE)
    add_library(gfx_runtime_metal_obj OBJECT)
    target_sources(gfx_runtime_metal_obj PRIVATE
        ${GFX_RUNTIME_COMMON_HEADERS}
        ${GFX_RUNTIME_METAL_SOURCES}
        ${GFX_RUNTIME_METAL_HEADERS})
    target_link_libraries(gfx_runtime_metal_obj PRIVATE
        gfx_plugin_build_settings
        openvino::runtime
        openvino::runtime::dev)
    set_target_properties(gfx_runtime_metal_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_sources(${TARGET_NAME} PRIVATE $<TARGET_OBJECTS:gfx_runtime_metal_obj>)
    if(DEFINED GFX_LLVM_EXTERNAL_TARGET)
        add_dependencies(gfx_runtime_metal_obj ${GFX_LLVM_EXTERNAL_TARGET})
    endif()
endif()

if(TARGET GFX::Metal)
    target_link_libraries(${TARGET_NAME} PRIVATE GFX::Metal)
endif()

if(GFX_BACKEND_VULKAN_AVAILABLE)
    add_library(gfx_runtime_vulkan_obj OBJECT)
    target_sources(gfx_runtime_vulkan_obj PRIVATE
        ${GFX_RUNTIME_COMMON_HEADERS}
        ${GFX_RUNTIME_VULKAN_SOURCES}
        ${GFX_RUNTIME_VULKAN_HEADERS})
    set(GFX_VULKAN_MLIR_LIBS
        MLIRSPIRVSerialization
        MLIRSPIRVTransforms
        MLIRSPIRVConversion
        MLIRGPUToSPIRV
        MLIRArithToSPIRV
        MLIRMathToSPIRV
        MLIRSCFToSPIRV
        MLIRTensorToSPIRV
        MLIRMemRefToSPIRV
        MLIRFuncToSPIRV
        MLIRIndexToSPIRV
        MLIRControlFlowToSPIRV
        MLIRVectorDialect
        MLIRVectorInterfaces
        MLIRMaskableOpInterface
        MLIRMaskingOpInterface
        MLIRSPIRVDialect
        MLIRSPIRVImageInterfaces
        MLIRSPIRVUtils
        MLIRSPIRVBinaryUtils
        MLIRUBDialect
        MLIRUBToSPIRV
        MLIRControlFlowDialect
        MLIRControlFlowInterfaces
        MLIRDLTIDialect
        MLIRIndexDialect
        MLIRMemRefDialect
        MLIRFunctionInterfaces
        MLIRDestinationStyleOpInterface
        MLIRShapedOpInterfaces
        MLIRIndexingMapOpInterface
        MLIRLoopLikeInterface
        MLIRParallelCombiningOpInterface
        MLIRMemorySlotInterfaces
        MLIRCopyOpInterface
        MLIRArithValueBoundsOpInterfaceImpl
        MLIRTensorInferTypeOpInterfaceImpl
        MLIRTensorTilingInterfaceImpl
        MLIRInferIntRangeCommon
        MLIRInferIntRangeInterface
        MLIRInferTypeOpInterface
        MLIRSideEffectInterfaces
        MLIRViewLikeInterface
        MLIRCastInterfaces
        MLIRDataLayoutInterfaces
        MLIRTransformUtils
        MLIRRewrite
        MLIRRewritePDL
        MLIRPDLToPDLInterp
        MLIRPDLInterpDialect
        MLIRPDLDialect
        MLIRPDLLAST
        MLIRSubsetOpInterface
        MLIRReconcileUnrealizedCasts
        MLIRParser
        MLIRAsmParser
        MLIRBytecodeReader
        MLIRBytecodeWriter
        MLIRBytecodeOpInterface
        MLIRDialect
        MLIRDialectUtils
        MLIRIR
        MLIRSupport
        MLIRPass
        MLIRTransforms
        MLIRAffineTransforms
        MLIRAffineToStandard
        MLIRAffineUtils
        MLIRAffineAnalysis
        MLIRAffineDialect
        MLIRTensorUtils
        MLIRTensorTransforms
        MLIRTensorTransformOps
        MLIRPresburger
        MLIRArithUtils
        MLIRSCFTransforms
        MLIRSCFUtils
        MLIRMemRefUtils
        MLIRVectorToSPIRV)
    set(GFX_VULKAN_MLIR_LINK_LIBS "")
    set(_gfx_mlir_lib_dir "")
    if(DEFINED MLIR_DIR)
        get_filename_component(_gfx_mlir_cmake_dir "${MLIR_DIR}" DIRECTORY)
        get_filename_component(_gfx_mlir_lib_dir "${_gfx_mlir_cmake_dir}" DIRECTORY)
    endif()
    foreach(_gfx_mlir_lib IN LISTS GFX_VULKAN_MLIR_LIBS)
        if(TARGET ${_gfx_mlir_lib})
            list(APPEND GFX_VULKAN_MLIR_LINK_LIBS ${_gfx_mlir_lib})
        elseif(_gfx_mlir_lib_dir)
            set(_gfx_mlir_lib_path "${_gfx_mlir_lib_dir}/lib${_gfx_mlir_lib}.a")
            if(EXISTS "${_gfx_mlir_lib_path}")
                list(APPEND GFX_VULKAN_MLIR_LINK_LIBS "${_gfx_mlir_lib_path}")
            else()
                list(APPEND GFX_VULKAN_MLIR_LINK_LIBS ${_gfx_mlir_lib})
            endif()
        else()
            list(APPEND GFX_VULKAN_MLIR_LINK_LIBS ${_gfx_mlir_lib})
        endif()
    endforeach()
    if(UNIX AND NOT APPLE)
        list(JOIN GFX_VULKAN_MLIR_LINK_LIBS "," GFX_VULKAN_MLIR_LIBS_JOINED)
        set(GFX_VULKAN_MLIR_LINK_LIBS "$<LINK_GROUP:RESCAN,${GFX_VULKAN_MLIR_LIBS_JOINED}>")
    endif()
    unset(_gfx_mlir_lib_dir)
    unset(_gfx_mlir_cmake_dir)
    unset(_gfx_mlir_lib_path)
    unset(_gfx_mlir_lib)
    add_library(gfx_vulkan_mlir_deps INTERFACE)
    target_link_libraries(gfx_vulkan_mlir_deps INTERFACE ${GFX_VULKAN_MLIR_LINK_LIBS})
    target_link_libraries(gfx_runtime_vulkan_obj PRIVATE
        gfx_plugin_build_settings
        openvino::runtime
        openvino::runtime::dev
        gfx_vulkan_mlir_deps)
    set_target_properties(gfx_runtime_vulkan_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_sources(${TARGET_NAME} PRIVATE $<TARGET_OBJECTS:gfx_runtime_vulkan_obj>)
    if(DEFINED GFX_LLVM_EXTERNAL_TARGET)
        add_dependencies(gfx_runtime_vulkan_obj ${GFX_LLVM_EXTERNAL_TARGET})
    endif()
    if(TARGET GFX::Vulkan)
        target_link_libraries(gfx_runtime_vulkan_obj PRIVATE GFX::Vulkan)
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_vulkan_mlir_deps)
endif()

if(TARGET GFX::Vulkan)
    target_link_libraries(${TARGET_NAME} PRIVATE GFX::Vulkan)
endif()

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
