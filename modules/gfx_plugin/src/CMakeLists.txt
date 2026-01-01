# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_gfx_plugin")

include(GNUInstallDirs)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GfxSources.cmake)

set(GFX_PLUGIN_COMMON_SOURCES ${GFX_PLUGIN_SOURCES})
set(GFX_PLUGIN_ENTRY_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/plugin/plugin.cpp")
list(FILTER GFX_PLUGIN_COMMON_SOURCES EXCLUDE REGEX "plugin/plugin\\.cpp$")
set(GFX_PLUGIN_METAL_SOURCES_EFFECTIVE "")
set(GFX_PLUGIN_VULKAN_SOURCES_EFFECTIVE "")
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

if(GFX_BACKEND_VULKAN_AVAILABLE)
    list(APPEND GFX_PLUGIN_VULKAN_SOURCES_EFFECTIVE
        ${GFX_PLUGIN_VULKAN_SOURCES})
    list(APPEND GFX_RUNTIME_VULKAN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/runtime/stage_factory.cpp)
else()
    list(APPEND GFX_PLUGIN_VULKAN_SOURCES_EFFECTIVE
        ${GFX_PLUGIN_VULKAN_STUB_SOURCES})
    list(APPEND GFX_RUNTIME_COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/runtime/stage_factory_stub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/runtime/op_support_stub.cpp
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

if(NOT GFX_PLUGIN_COMMON_SOURCES)
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

# Build common runtime as a separate static library (core target).
add_library(gfx_runtime_common STATIC)
target_sources(gfx_runtime_common PRIVATE
    ${GFX_RUNTIME_COMMON_HEADERS}
    ${GFX_RUNTIME_COMMON_SOURCES})
target_link_libraries(gfx_runtime_common PRIVATE
    gfx_plugin_build_settings
    openvino::runtime
    openvino::runtime::dev)
set_target_properties(gfx_runtime_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ENABLE_GFX_MLIR)
    add_library(gfx_runtime_mlir STATIC)
    target_sources(gfx_runtime_mlir PRIVATE
        ${GFX_RUNTIME_MLIR_HEADERS}
        ${GFX_RUNTIME_MLIR_SOURCES})
    target_link_libraries(gfx_runtime_mlir PRIVATE
        gfx_plugin_build_settings
        openvino::runtime
        openvino::runtime::dev)
    set_target_properties(gfx_runtime_mlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(DEFINED GFX_LLVM_EXTERNAL_TARGET)
        add_dependencies(gfx_runtime_mlir ${GFX_LLVM_EXTERNAL_TARGET})
    endif()
endif()

# Build common plugin sources as a separate static library (core target).
add_library(gfx_plugin_core STATIC)
target_sources(gfx_plugin_core PRIVATE
    ${GFX_PLUGIN_COMMON_SOURCES}
    ${GFX_PLUGIN_HEADERS})
target_link_libraries(gfx_plugin_core PRIVATE
    gfx_plugin_build_settings
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
    openvino::runtime
    openvino::runtime::dev)
set_target_properties(gfx_plugin_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(gfx_plugin_vulkan STATIC)
target_sources(gfx_plugin_vulkan PRIVATE
    ${GFX_PLUGIN_VULKAN_SOURCES_EFFECTIVE}
    ${GFX_PLUGIN_VULKAN_HEADERS})
target_link_libraries(gfx_plugin_vulkan PRIVATE
    gfx_plugin_build_settings
    openvino::runtime
    openvino::runtime::dev)
set_target_properties(gfx_plugin_vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Build GFX plugin as SHARED (not MODULE) to allow normal linking/registration in OpenVINO
add_library(${TARGET_NAME} SHARED)
target_sources(${TARGET_NAME} PRIVATE
    ${GFX_PLUGIN_HEADERS}
    ${GFX_PLUGIN_ENTRY_SOURCE})
target_link_libraries(${TARGET_NAME} PRIVATE gfx_plugin_build_settings)
target_link_libraries(${TARGET_NAME} PRIVATE
    gfx_plugin_core
    gfx_plugin_metal
    gfx_plugin_vulkan
    gfx_runtime_common
    $<$<BOOL:${ENABLE_GFX_MLIR}>:gfx_runtime_mlir>)

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
    $<$<BOOL:${ENABLE_GFX_MLIR}>:gfx_mlir>
    $<$<BOOL:${ENABLE_GFX_MLIR}>:gfx_mlir_codegen>)

if(GFX_BACKEND_METAL_AVAILABLE)
    add_library(gfx_runtime_metal STATIC)
    target_sources(gfx_runtime_metal PRIVATE
        ${GFX_RUNTIME_COMMON_HEADERS}
        ${GFX_RUNTIME_METAL_SOURCES}
        ${GFX_RUNTIME_METAL_HEADERS})
    target_link_libraries(gfx_runtime_metal PRIVATE
        gfx_plugin_build_settings
        openvino::runtime
        openvino::runtime::dev)
    set_target_properties(gfx_runtime_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_runtime_metal)
    if(DEFINED GFX_LLVM_EXTERNAL_TARGET)
        add_dependencies(gfx_runtime_metal ${GFX_LLVM_EXTERNAL_TARGET})
    endif()
endif()

if(TARGET GFX::Metal)
    target_link_libraries(${TARGET_NAME} PRIVATE GFX::Metal)
endif()

if(GFX_BACKEND_VULKAN_AVAILABLE)
    add_library(gfx_runtime_vulkan STATIC)
    target_sources(gfx_runtime_vulkan PRIVATE
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
    target_link_libraries(gfx_runtime_vulkan PRIVATE
        gfx_plugin_build_settings
        openvino::runtime
        openvino::runtime::dev
        gfx_vulkan_mlir_deps)
    set_target_properties(gfx_runtime_vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_runtime_vulkan)
    if(DEFINED GFX_LLVM_EXTERNAL_TARGET)
        add_dependencies(gfx_runtime_vulkan ${GFX_LLVM_EXTERNAL_TARGET})
    endif()
    if(TARGET GFX::Vulkan)
        target_link_libraries(gfx_runtime_vulkan PRIVATE GFX::Vulkan)
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

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/openvino/gfx_plugin
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvino
        COMPONENT gfx)
