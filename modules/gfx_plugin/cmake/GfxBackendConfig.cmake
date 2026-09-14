# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Configure available backends and default selection at configure time.

include_guard(GLOBAL)

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)

set(GFX_DEFAULT_BACKEND "auto" CACHE STRING "Default GFX backend (auto, metal, opencl)")
set_property(CACHE GFX_DEFAULT_BACKEND PROPERTY STRINGS auto metal opencl)

option(GFX_ENABLE_METAL "Enable GFX Metal backend if available" ON)
option(GFX_ENABLE_OPENCL "Enable GFX OpenCL source-kernel backend if available" ON)

if(APPLE)
    if(GFX_ENABLE_OPENCL)
        message(STATUS "GFX: OpenCL backend disabled on macOS; Apple path is Metal/MPS/MPSRT/MSL.")
    endif()
    set(GFX_ENABLE_OPENCL OFF CACHE BOOL "Enable GFX OpenCL source-kernel backend if available" FORCE)
endif()

if(NOT DEFINED GFX_HAS_METAL_SOURCES)
    set(GFX_HAS_METAL_SOURCES ON)
endif()

if(NOT DEFINED GFX_HAS_OPENCL_SOURCES)
    set(GFX_HAS_OPENCL_SOURCES OFF)
endif()

set(GFX_BACKEND_METAL_AVAILABLE OFF)
set(GFX_BACKEND_OPENCL_AVAILABLE OFF)
set(GFX_METAL_LIBRARIES "")
set(GFX_METAL_INCLUDE_DIRS "")
set(GFX_OPENCL_LIBRARIES "")
set(GFX_OPENCL_INCLUDE_DIRS "")

function(_gfx_check_metal_backend out_var)
    set(${out_var} OFF PARENT_SCOPE)

    if(NOT APPLE)
        return()
    endif()

    foreach(_gfx_metal_cache_var
            GFX_METAL_LIBRARY
            GFX_METAL_PERFORMANCE_SHADERS_LIBRARY
            GFX_FOUNDATION_LIBRARY
            GFX_METAL_INCLUDE_DIR)
        if(DEFINED ${_gfx_metal_cache_var} AND NOT EXISTS "${${_gfx_metal_cache_var}}")
            unset(${_gfx_metal_cache_var} CACHE)
        endif()
    endforeach()

    unset(GFX_METAL_COMPILES CACHE)

    set(_gfx_framework_paths
        "${CMAKE_OSX_SYSROOT}/System/Library/Frameworks"
        "/System/Library/Frameworks")

    find_library(GFX_METAL_LIBRARY Metal PATHS ${_gfx_framework_paths})
    find_library(GFX_METAL_PERFORMANCE_SHADERS_LIBRARY MetalPerformanceShaders PATHS ${_gfx_framework_paths})
    find_library(GFX_METAL_PERFORMANCE_SHADERS_GRAPH_LIBRARY MetalPerformanceShadersGraph
        PATHS ${_gfx_framework_paths})
    find_library(GFX_FOUNDATION_LIBRARY Foundation PATHS ${_gfx_framework_paths})
    find_path(GFX_METAL_INCLUDE_DIR Metal/Metal.h
        PATHS ${_gfx_framework_paths}
        PATH_SUFFIXES Metal.framework/Headers)

    if(NOT GFX_METAL_LIBRARY OR NOT GFX_METAL_PERFORMANCE_SHADERS_LIBRARY OR
       NOT GFX_METAL_PERFORMANCE_SHADERS_GRAPH_LIBRARY OR
       NOT GFX_FOUNDATION_LIBRARY OR NOT GFX_METAL_INCLUDE_DIR)
        return()
    endif()

    cmake_push_check_state(RESET)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_REQUIRED_INCLUDES "${GFX_METAL_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES
        "${GFX_METAL_LIBRARY};${GFX_METAL_PERFORMANCE_SHADERS_LIBRARY};${GFX_FOUNDATION_LIBRARY}")
    set(CMAKE_REQUIRED_FLAGS "-x objective-c++")
    set(_gfx_metal_test [=[
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
int main() {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  (void)MPSSupportsMTLDevice(dev);
  MPSGraph* graph = [[MPSGraph alloc] init];
  (void)graph;
  return 0;
}
]=])
    check_cxx_source_compiles("${_gfx_metal_test}" GFX_METAL_COMPILES)
    cmake_pop_check_state()

    if(GFX_METAL_COMPILES)
        set(${out_var} ON PARENT_SCOPE)
        set(GFX_METAL_LIBRARIES
            "${GFX_METAL_LIBRARY};${GFX_METAL_PERFORMANCE_SHADERS_LIBRARY};${GFX_METAL_PERFORMANCE_SHADERS_GRAPH_LIBRARY};${GFX_FOUNDATION_LIBRARY}"
            PARENT_SCOPE)
        set(GFX_METAL_INCLUDE_DIRS "${GFX_METAL_INCLUDE_DIR}" PARENT_SCOPE)
    endif()

    unset(_gfx_framework_paths)
endfunction()

if(GFX_ENABLE_METAL AND GFX_HAS_METAL_SOURCES)
    _gfx_check_metal_backend(GFX_BACKEND_METAL_AVAILABLE)
elseif(GFX_ENABLE_METAL AND NOT GFX_HAS_METAL_SOURCES)
    message(STATUS "GFX: Metal backend sources not present; disabling Metal backend.")
endif()

if(GFX_ENABLE_OPENCL AND GFX_HAS_OPENCL_SOURCES)
    # The target OpenCL runtime is loaded dynamically by the backend. Compile-time
    # availability means the source-kernel backend is present in this build.
    set(GFX_BACKEND_OPENCL_AVAILABLE ON)
elseif(GFX_ENABLE_OPENCL AND NOT GFX_HAS_OPENCL_SOURCES)
    message(STATUS "GFX: OpenCL backend sources not present; disabling OpenCL backend.")
endif()

if(GFX_BACKEND_METAL_AVAILABLE AND NOT TARGET GFX::Metal)
    add_library(GFX::Metal INTERFACE IMPORTED GLOBAL)
    set_target_properties(GFX::Metal PROPERTIES
        INTERFACE_LINK_LIBRARIES "${GFX_METAL_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${GFX_METAL_INCLUDE_DIRS}")
endif()

set(GFX_RESOLVED_DEFAULT_BACKEND "")

if(GFX_DEFAULT_BACKEND STREQUAL "auto")
    if(GFX_BACKEND_METAL_AVAILABLE)
        set(GFX_RESOLVED_DEFAULT_BACKEND "metal")
    elseif(GFX_BACKEND_OPENCL_AVAILABLE)
        set(GFX_RESOLVED_DEFAULT_BACKEND "opencl")
    else()
        message(FATAL_ERROR "GFX_DEFAULT_BACKEND=auto could not resolve a backend: Metal and OpenCL are unavailable.")
    endif()
elseif(GFX_DEFAULT_BACKEND STREQUAL "metal")
    if(NOT GFX_BACKEND_METAL_AVAILABLE)
        message(FATAL_ERROR "GFX_DEFAULT_BACKEND=metal was requested, but the Metal backend is unavailable in this build.")
    endif()
    set(GFX_RESOLVED_DEFAULT_BACKEND "metal")
elseif(GFX_DEFAULT_BACKEND STREQUAL "opencl")
    if(NOT GFX_BACKEND_OPENCL_AVAILABLE)
        message(FATAL_ERROR "GFX_DEFAULT_BACKEND=opencl was requested, but the OpenCL backend is unavailable in this build.")
    endif()
    set(GFX_RESOLVED_DEFAULT_BACKEND "opencl")
else()
    message(FATAL_ERROR "GFX_DEFAULT_BACKEND must be one of: auto, metal, opencl")
endif()

message(STATUS "GFX backends: metal=${GFX_BACKEND_METAL_AVAILABLE} (sources=${GFX_HAS_METAL_SOURCES}), opencl=${GFX_BACKEND_OPENCL_AVAILABLE} (sources=${GFX_HAS_OPENCL_SOURCES}), default=${GFX_RESOLVED_DEFAULT_BACKEND}, requested=${GFX_DEFAULT_BACKEND}")

if(GFX_BACKEND_METAL_AVAILABLE)
    set(GFX_BACKEND_METAL_AVAILABLE_VALUE 1)
else()
    set(GFX_BACKEND_METAL_AVAILABLE_VALUE 0)
endif()

if(GFX_BACKEND_OPENCL_AVAILABLE)
    set(GFX_BACKEND_OPENCL_AVAILABLE_VALUE 1)
else()
    set(GFX_BACKEND_OPENCL_AVAILABLE_VALUE 0)
endif()

# Keep cache in sync with computed availability to avoid stale values on reconfigure.
set(GFX_BACKEND_METAL_AVAILABLE ${GFX_BACKEND_METAL_AVAILABLE} CACHE BOOL "Enable GFX Metal backend" FORCE)
set(GFX_BACKEND_OPENCL_AVAILABLE ${GFX_BACKEND_OPENCL_AVAILABLE} CACHE BOOL "Enable GFX OpenCL backend" FORCE)
