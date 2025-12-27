# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Configure available backends and default selection at configure time.

include_guard(GLOBAL)

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)

set(GFX_DEFAULT_BACKEND "auto" CACHE STRING "Default GFX backend (auto, metal, vulkan)")
set_property(CACHE GFX_DEFAULT_BACKEND PROPERTY STRINGS auto metal vulkan)

option(GFX_ENABLE_METAL "Enable GFX Metal backend if available" ON)
option(GFX_ENABLE_VULKAN "Enable GFX Vulkan backend if available" ON)

if(APPLE)
    if(GFX_ENABLE_VULKAN)
        message(STATUS "GFX: Vulkan backend disabled on macOS")
    endif()
    set(GFX_ENABLE_VULKAN OFF CACHE BOOL "Enable GFX Vulkan backend if available" FORCE)
endif()

if(NOT DEFINED GFX_HAS_METAL_SOURCES)
    set(GFX_HAS_METAL_SOURCES ON)
endif()

if(NOT DEFINED GFX_HAS_VULKAN_SOURCES)
    set(GFX_HAS_VULKAN_SOURCES ON)
endif()

set(GFX_BACKEND_METAL_AVAILABLE OFF)
set(GFX_BACKEND_VULKAN_AVAILABLE OFF)
set(GFX_METAL_LIBRARIES "")
set(GFX_METAL_INCLUDE_DIRS "")
set(GFX_VULKAN_LIBRARIES "")
set(GFX_VULKAN_INCLUDE_DIRS "")

function(_gfx_check_metal_backend out_var)
    set(${out_var} OFF PARENT_SCOPE)

    if(NOT APPLE)
        return()
    endif()

    unset(GFX_METAL_COMPILES CACHE)

    set(_gfx_framework_paths
        "${CMAKE_OSX_SYSROOT}/System/Library/Frameworks"
        "/System/Library/Frameworks")

    find_library(GFX_METAL_LIBRARY Metal PATHS ${_gfx_framework_paths})
    find_library(GFX_FOUNDATION_LIBRARY Foundation PATHS ${_gfx_framework_paths})
    find_path(GFX_METAL_INCLUDE_DIR Metal/Metal.h
        PATHS ${_gfx_framework_paths}
        PATH_SUFFIXES Metal.framework/Headers)

    if(NOT GFX_METAL_LIBRARY OR NOT GFX_FOUNDATION_LIBRARY OR NOT GFX_METAL_INCLUDE_DIR)
        return()
    endif()

    cmake_push_check_state(RESET)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_REQUIRED_INCLUDES "${GFX_METAL_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES "${GFX_METAL_LIBRARY};${GFX_FOUNDATION_LIBRARY}")
    set(CMAKE_REQUIRED_FLAGS "-x objective-c++")
    set(_gfx_metal_test [=[
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
int main() {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  (void)dev;
  return 0;
}
]=])
    check_cxx_source_compiles("${_gfx_metal_test}" GFX_METAL_COMPILES)
    cmake_pop_check_state()

    if(GFX_METAL_COMPILES)
        set(${out_var} ON PARENT_SCOPE)
        set(GFX_METAL_LIBRARIES "${GFX_METAL_LIBRARY};${GFX_FOUNDATION_LIBRARY}" PARENT_SCOPE)
        set(GFX_METAL_INCLUDE_DIRS "${GFX_METAL_INCLUDE_DIR}" PARENT_SCOPE)
    endif()

    unset(_gfx_framework_paths)
endfunction()

function(_gfx_check_vulkan_backend out_var)
    set(${out_var} OFF PARENT_SCOPE)

    unset(GFX_VULKAN_COMPILES CACHE)

    set(_gfx_vulkan_libs "")
    set(_gfx_vulkan_includes "")

    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        set(_gfx_vulkan_libs Vulkan::Vulkan)
        if(DEFINED Vulkan_INCLUDE_DIRS)
            set(_gfx_vulkan_includes "${Vulkan_INCLUDE_DIRS}")
        elseif(DEFINED Vulkan_INCLUDE_DIR)
            set(_gfx_vulkan_includes "${Vulkan_INCLUDE_DIR}")
        endif()
    else()
        set(_gfx_vulkan_hint_paths "")
        set(_gfx_vulkan_lib_paths "")
        if(DEFINED ENV{VULKAN_SDK})
            list(APPEND _gfx_vulkan_hint_paths
                "$ENV{VULKAN_SDK}/Include"
                "$ENV{VULKAN_SDK}/include")
            list(APPEND _gfx_vulkan_lib_paths
                "$ENV{VULKAN_SDK}/Lib"
                "$ENV{VULKAN_SDK}/lib")
        endif()

        find_library(GFX_VULKAN_LIBRARY NAMES vulkan vulkan-1 MoltenVK
            HINTS ${_gfx_vulkan_lib_paths})
        find_path(GFX_VULKAN_INCLUDE_DIR vulkan/vulkan.h
            HINTS ${_gfx_vulkan_hint_paths})
        if(GFX_VULKAN_LIBRARY AND GFX_VULKAN_INCLUDE_DIR)
            set(_gfx_vulkan_libs "${GFX_VULKAN_LIBRARY}")
            set(_gfx_vulkan_includes "${GFX_VULKAN_INCLUDE_DIR}")
        endif()

        unset(_gfx_vulkan_hint_paths)
        unset(_gfx_vulkan_lib_paths)
    endif()

    if(NOT _gfx_vulkan_libs OR NOT _gfx_vulkan_includes)
        return()
    endif()

    cmake_push_check_state(RESET)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_REQUIRED_INCLUDES "${_gfx_vulkan_includes}")
    set(CMAKE_REQUIRED_LIBRARIES "${_gfx_vulkan_libs}")
    set(_gfx_vulkan_test [=[
#include <vulkan/vulkan.h>
int main() {
  VkInstance inst = VK_NULL_HANDLE;
  VkInstanceCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  VkResult res = vkCreateInstance(&info, 0, &inst);
  if (inst) {
    vkDestroyInstance(inst, 0);
  }
  return res == VK_SUCCESS ? 0 : 0;
}
]=])
    check_cxx_source_compiles("${_gfx_vulkan_test}" GFX_VULKAN_COMPILES)
    cmake_pop_check_state()

    if(GFX_VULKAN_COMPILES)
        set(${out_var} ON PARENT_SCOPE)
        set(GFX_VULKAN_LIBRARIES "${_gfx_vulkan_libs}" PARENT_SCOPE)
        set(GFX_VULKAN_INCLUDE_DIRS "${_gfx_vulkan_includes}" PARENT_SCOPE)
    endif()
endfunction()

if(GFX_ENABLE_METAL AND GFX_HAS_METAL_SOURCES)
    _gfx_check_metal_backend(GFX_BACKEND_METAL_AVAILABLE)
elseif(GFX_ENABLE_METAL AND NOT GFX_HAS_METAL_SOURCES)
    message(STATUS "GFX: Metal backend sources not present; disabling Metal backend.")
endif()

if(GFX_ENABLE_VULKAN AND GFX_HAS_VULKAN_SOURCES)
    _gfx_check_vulkan_backend(GFX_BACKEND_VULKAN_AVAILABLE)
elseif(GFX_ENABLE_VULKAN AND NOT GFX_HAS_VULKAN_SOURCES)
    message(STATUS "GFX: Vulkan backend sources not present; disabling Vulkan backend.")
endif()

if(GFX_BACKEND_METAL_AVAILABLE AND NOT TARGET GFX::Metal)
    add_library(GFX::Metal INTERFACE IMPORTED GLOBAL)
    set_target_properties(GFX::Metal PROPERTIES
        INTERFACE_LINK_LIBRARIES "${GFX_METAL_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${GFX_METAL_INCLUDE_DIRS}")
endif()

if(GFX_BACKEND_VULKAN_AVAILABLE AND NOT TARGET GFX::Vulkan)
    add_library(GFX::Vulkan INTERFACE IMPORTED GLOBAL)
    if(TARGET Vulkan::Vulkan)
        set(_gfx_vulkan_link Vulkan::Vulkan)
    else()
        set(_gfx_vulkan_link "${GFX_VULKAN_LIBRARIES}")
    endif()
    set_target_properties(GFX::Vulkan PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_gfx_vulkan_link}"
        INTERFACE_INCLUDE_DIRECTORIES "${GFX_VULKAN_INCLUDE_DIRS}")
endif()

set(GFX_RESOLVED_DEFAULT_BACKEND "")

if(GFX_DEFAULT_BACKEND STREQUAL "auto")
    if(GFX_BACKEND_METAL_AVAILABLE)
        set(GFX_RESOLVED_DEFAULT_BACKEND "metal")
    elseif(GFX_BACKEND_VULKAN_AVAILABLE)
        set(GFX_RESOLVED_DEFAULT_BACKEND "vulkan")
    else()
        set(GFX_RESOLVED_DEFAULT_BACKEND "metal")
    endif()
elseif(GFX_DEFAULT_BACKEND STREQUAL "metal")
    if(GFX_BACKEND_METAL_AVAILABLE)
        set(GFX_RESOLVED_DEFAULT_BACKEND "metal")
    elseif(GFX_BACKEND_VULKAN_AVAILABLE)
        message(WARNING "GFX: Metal backend requested but unavailable; falling back to Vulkan.")
        set(GFX_RESOLVED_DEFAULT_BACKEND "vulkan")
    else()
        message(WARNING "GFX: Metal backend requested but unavailable; defaulting to metal anyway.")
        set(GFX_RESOLVED_DEFAULT_BACKEND "metal")
    endif()
elseif(GFX_DEFAULT_BACKEND STREQUAL "vulkan")
    if(GFX_BACKEND_VULKAN_AVAILABLE)
        set(GFX_RESOLVED_DEFAULT_BACKEND "vulkan")
    elseif(GFX_BACKEND_METAL_AVAILABLE)
        message(WARNING "GFX: Vulkan backend requested but unavailable; falling back to Metal.")
        set(GFX_RESOLVED_DEFAULT_BACKEND "metal")
    else()
        message(WARNING "GFX: Vulkan backend requested but unavailable; defaulting to vulkan anyway.")
        set(GFX_RESOLVED_DEFAULT_BACKEND "vulkan")
    endif()
else()
    message(FATAL_ERROR "GFX_DEFAULT_BACKEND must be one of: auto, metal, vulkan")
endif()

message(STATUS "GFX backends: metal=${GFX_BACKEND_METAL_AVAILABLE} (sources=${GFX_HAS_METAL_SOURCES}), vulkan=${GFX_BACKEND_VULKAN_AVAILABLE} (sources=${GFX_HAS_VULKAN_SOURCES}), default=${GFX_RESOLVED_DEFAULT_BACKEND}, requested=${GFX_DEFAULT_BACKEND}")

if(GFX_BACKEND_METAL_AVAILABLE)
    set(GFX_BACKEND_METAL_AVAILABLE_VALUE 1)
else()
    set(GFX_BACKEND_METAL_AVAILABLE_VALUE 0)
endif()

if(GFX_BACKEND_VULKAN_AVAILABLE)
    set(GFX_BACKEND_VULKAN_AVAILABLE_VALUE 1)
else()
    set(GFX_BACKEND_VULKAN_AVAILABLE_VALUE 0)
endif()

# Keep cache in sync with computed availability to avoid stale values on reconfigure.
set(GFX_BACKEND_METAL_AVAILABLE ${GFX_BACKEND_METAL_AVAILABLE} CACHE BOOL "Enable GFX Metal backend" FORCE)
set(GFX_BACKEND_VULKAN_AVAILABLE ${GFX_BACKEND_VULKAN_AVAILABLE} CACHE BOOL "Enable GFX Vulkan backend" FORCE)
