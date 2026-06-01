# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(_gfx_raspberry_opencl_default OFF)
if(GFX_BACKEND_OPENCL_AVAILABLE AND
   CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   NOT ANDROID AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|armv7|armv7l|ARM)$")
    set(_gfx_raspberry_opencl_default ON)
endif()

option(GFX_ENABLE_RASPBERRY_OPENCL_TOOLCHAIN
       "Build and bundle CLVK/CLSPV OpenCL runtime for Raspberry Linux OpenCL targets"
       ${_gfx_raspberry_opencl_default})

set(GFX_RASPBERRY_CLVK_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/clvk"
    CACHE PATH "CLVK source directory used by the Raspberry OpenCL backend")
set(GFX_RASPBERRY_CLSPV_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/clspv"
    CACHE PATH "CLSPV source directory used by the Raspberry OpenCL backend")
set(GFX_RASPBERRY_OPENCL_BUILD_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/third_party/raspberry-opencl/clvk"
    CACHE PATH "CLVK/CLSPV build directory used by the Raspberry OpenCL backend")
if(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(_gfx_raspberry_opencl_default_bundle_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/opencl")
else()
    set(_gfx_raspberry_opencl_default_bundle_dir "${CMAKE_CURRENT_BINARY_DIR}/opencl")
endif()
set(GFX_RASPBERRY_OPENCL_BUNDLE_DIR
    "${_gfx_raspberry_opencl_default_bundle_dir}"
    CACHE PATH "Directory where the Raspberry OpenCL runtime bundle is staged")

if(NOT GFX_ENABLE_RASPBERRY_OPENCL_TOOLCHAIN)
    unset(_gfx_raspberry_opencl_default_bundle_dir)
    unset(_gfx_raspberry_opencl_default)
    return()
endif()

if(NOT GFX_BACKEND_OPENCL_AVAILABLE)
    message(FATAL_ERROR "GFX: Raspberry OpenCL toolchain requires the shared OpenCL backend to be available.")
endif()

foreach(_gfx_required_dir IN ITEMS
        "${GFX_RASPBERRY_CLVK_SOURCE_DIR}"
        "${GFX_RASPBERRY_CLSPV_SOURCE_DIR}"
        "${GFX_LLVM_PROJECT_SOURCE_DIR}/llvm"
        "${GFX_LLVM_PROJECT_SOURCE_DIR}/clang"
        "${GFX_LLVM_PROJECT_SOURCE_DIR}/libclc")
    if(NOT EXISTS "${_gfx_required_dir}/CMakeLists.txt")
        message(FATAL_ERROR
            "GFX: missing Raspberry OpenCL third-party source: ${_gfx_required_dir}. "
            "Update submodules recursively before building Raspberry targets.")
    endif()
endforeach()

foreach(_gfx_required_dir IN ITEMS
        "${GFX_RASPBERRY_CLVK_SOURCE_DIR}/external/OpenCL-Headers"
        "${GFX_RASPBERRY_CLVK_SOURCE_DIR}/external/SPIRV-Headers"
        "${GFX_RASPBERRY_CLVK_SOURCE_DIR}/external/SPIRV-Tools")
    if(NOT EXISTS "${_gfx_required_dir}/CMakeLists.txt")
        message(FATAL_ERROR
            "GFX: missing CLVK dependency source: ${_gfx_required_dir}. "
            "Run: git -C modules/gfx_plugin/third_party/clvk submodule update --init "
            "external/OpenCL-Headers external/SPIRV-Headers external/SPIRV-Tools")
    endif()
endforeach()

file(MAKE_DIRECTORY "${GFX_RASPBERRY_OPENCL_BUNDLE_DIR}")

if(CMAKE_C_COMPILER_TARGET)
    set(_gfx_raspberry_opencl_target_triple "${CMAKE_C_COMPILER_TARGET}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_gfx_raspberry_opencl_target_triple "aarch64-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|armv7l|ARM)$")
    set(_gfx_raspberry_opencl_target_triple "arm-linux-gnueabihf")
else()
    message(FATAL_ERROR
        "GFX: cannot infer Raspberry OpenCL target triple for ${CMAKE_SYSTEM_PROCESSOR}. "
        "Set CMAKE_C_COMPILER_TARGET or disable GFX_ENABLE_RASPBERRY_OPENCL_TOOLCHAIN.")
endif()

get_filename_component(_gfx_raspberry_opencl_cross_bin_dir
                       "${CMAKE_C_COMPILER}" DIRECTORY)
get_filename_component(_gfx_raspberry_opencl_toolchain_dir
                       "${_gfx_raspberry_opencl_cross_bin_dir}" DIRECTORY)
set(GFX_RASPBERRY_HOST_LLVM_TOOLS_DIR
    "${_gfx_raspberry_opencl_toolchain_dir}/host-llvm-build/bin"
    CACHE PATH "Host LLVM tools used to generate CLSPV/libclc builtin libraries")
if(EXISTS "${GFX_RASPBERRY_HOST_LLVM_TOOLS_DIR}/clang")
    set(_gfx_raspberry_host_clang "${GFX_RASPBERRY_HOST_LLVM_TOOLS_DIR}/clang")
elseif(EXISTS "${GFX_RASPBERRY_HOST_LLVM_TOOLS_DIR}/clang-22")
    set(_gfx_raspberry_host_clang "${GFX_RASPBERRY_HOST_LLVM_TOOLS_DIR}/clang-22")
else()
    message(FATAL_ERROR
        "GFX: missing host clang in ${GFX_RASPBERRY_HOST_LLVM_TOOLS_DIR}; "
        "CLSPV/libclc generation requires host LLVM tools from the Raspberry toolchain.")
endif()
foreach(_gfx_required_tool IN ITEMS llvm-as llvm-link opt)
    if(NOT EXISTS "${GFX_RASPBERRY_HOST_LLVM_TOOLS_DIR}/${_gfx_required_tool}")
        message(FATAL_ERROR
            "GFX: missing host LLVM tool ${_gfx_required_tool} in "
            "${GFX_RASPBERRY_HOST_LLVM_TOOLS_DIR}; CLSPV/libclc generation requires it.")
    endif()
endforeach()

set(_gfx_raspberry_opencl_runtimes_args
    -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
    -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
    -DCMAKE_C_COMPILER_TARGET=${_gfx_raspberry_opencl_target_triple}
    -DCMAKE_CXX_COMPILER_TARGET=${_gfx_raspberry_opencl_target_triple}
    -DCMAKE_ASM_COMPILER_TARGET=${_gfx_raspberry_opencl_target_triple}
    -DCLANG=${_gfx_raspberry_host_clang}
    -DLLVM_AS=${GFX_RASPBERRY_HOST_LLVM_TOOLS_DIR}/llvm-as
    -DLLVM_LINK=${GFX_RASPBERRY_HOST_LLVM_TOOLS_DIR}/llvm-link
    -DOPT=${GFX_RASPBERRY_HOST_LLVM_TOOLS_DIR}/opt)
if(CMAKE_SYSROOT)
    list(APPEND _gfx_raspberry_opencl_runtimes_args
        -DCMAKE_SYSROOT=${CMAKE_SYSROOT})
endif()
string(REPLACE ";" "|" _gfx_raspberry_opencl_runtimes_args_arg
       "${_gfx_raspberry_opencl_runtimes_args}")

set(_gfx_raspberry_opencl_cmake_args
    -G ${CMAKE_GENERATOR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCLVK_BUILD_TESTS=OFF
    -DCLVK_BUILD_STATIC_TESTS=OFF
    -DCLVK_UNIT_TESTING=OFF
    -DCLVK_VULKAN_IMPLEMENTATION=system
    -DCLVK_COMPILER_AVAILABLE=ON
    -DCLVK_CLSPV_ONLINE_COMPILER=OFF
    -DCLVK_ENABLE_SPIRV_IL=OFF
    -DCLVK_BUILD_SPIRV_TOOLS=ON
    -DCLSPV_SOURCE_DIR=${GFX_RASPBERRY_CLSPV_SOURCE_DIR}
    -DCLSPV_BUILD_TESTS=OFF
    -DENABLE_CLSPV_OPT=OFF
    -DCLSPV_LLVM_SOURCE_DIR=${GFX_LLVM_PROJECT_SOURCE_DIR}/llvm
    -DCLSPV_CLANG_SOURCE_DIR=${GFX_LLVM_PROJECT_SOURCE_DIR}/clang
    -DCLSPV_LIBCLC_SOURCE_DIR=${GFX_LLVM_PROJECT_SOURCE_DIR}/libclc
    -DLLVM_HOST_TRIPLE=${_gfx_raspberry_opencl_target_triple}
    -DLLVM_DEFAULT_TARGET_TRIPLE=${_gfx_raspberry_opencl_target_triple}
    -DRUNTIMES_CMAKE_ARGS=${_gfx_raspberry_opencl_runtimes_args_arg}
    -DCLANG_BUILD_TOOLS=OFF
    -DCLANG_ENABLE_ARCMT=OFF
    -DCLANG_ENABLE_STATIC_ANALYZER=OFF
    -DLLVM_ENABLE_BINDINGS=OFF
    -DLLVM_BUILD_TOOLS=OFF
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_DOCS=OFF
    -DLLVM_INCLUDE_EXAMPLES=OFF
    # CLSPV generates builtin libraries through LLVM's libclc runtime targets.
    -DLLVM_INCLUDE_RUNTIMES=ON
    -DLLVM_INCLUDE_TESTS=OFF
    # LLVM runtimes CMake references llvm-config even when tool binaries are not
    # part of the requested build target.
    -DLLVM_INCLUDE_TOOLS=ON
    -DLLVM_INCLUDE_UTILS=OFF
    -DCLSPV_BUILD_SPIRV_DIS=OFF)

if(DEFINED _gfx_llvm_toolchain_args)
    list(APPEND _gfx_raspberry_opencl_cmake_args ${_gfx_llvm_toolchain_args})
endif()
if(DEFINED _gfx_llvm_optional_deps_off)
    list(APPEND _gfx_raspberry_opencl_cmake_args ${_gfx_llvm_optional_deps_off})
endif()

ExternalProject_Add(gfx_raspberry_opencl_toolchain
    SOURCE_DIR ${GFX_RASPBERRY_CLVK_SOURCE_DIR}
    BINARY_DIR ${GFX_RASPBERRY_OPENCL_BUILD_DIR}
    LIST_SEPARATOR |
    CMAKE_ARGS
        ${_gfx_raspberry_opencl_cmake_args}
    BUILD_COMMAND
        ${CMAKE_COMMAND} --build <BINARY_DIR> --target OpenCL clspv
    INSTALL_COMMAND
        ${CMAKE_COMMAND}
            -DCLVK_BUILD_DIR=<BINARY_DIR>
            -DOUTPUT_DIR=${GFX_RASPBERRY_OPENCL_BUNDLE_DIR}
            -P ${CMAKE_CURRENT_LIST_DIR}/InstallRaspberryOpenCLBundle.cmake
    BUILD_BYPRODUCTS
        "${GFX_RASPBERRY_OPENCL_BUNDLE_DIR}/libOpenCL.so.0.1"
        "${GFX_RASPBERRY_OPENCL_BUNDLE_DIR}/clspv")

set(GFX_RASPBERRY_OPENCL_BUNDLE_TARGET
    gfx_raspberry_opencl_toolchain
    CACHE INTERNAL "Target that builds and stages the Raspberry CLVK/CLSPV OpenCL runtime bundle")

unset(_gfx_required_dir)
unset(_gfx_required_tool)
unset(_gfx_raspberry_opencl_cmake_args)
unset(_gfx_raspberry_opencl_cross_bin_dir)
unset(_gfx_raspberry_opencl_default_bundle_dir)
unset(_gfx_raspberry_opencl_default)
unset(_gfx_raspberry_opencl_toolchain_dir)
unset(_gfx_raspberry_opencl_runtimes_args)
unset(_gfx_raspberry_opencl_runtimes_args_arg)
unset(_gfx_raspberry_opencl_target_triple)
unset(_gfx_raspberry_host_clang)
