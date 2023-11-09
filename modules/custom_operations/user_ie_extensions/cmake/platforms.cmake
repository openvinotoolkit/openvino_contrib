
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_CL_64)
  set(MSVC64 ON)
endif()

if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine
                  OUTPUT_VARIABLE OPENVINO_GCC_TARGET_MACHINE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(OPENVINO_GCC_TARGET_MACHINE MATCHES "amd64|x86_64|AMD64")
    set(MINGW64 ON)
  endif()
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(OV_HOST_ARCH X86_64)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*|amd64.*|AMD64.*")
  set(OV_HOST_ARCH X86)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*)")
  set(OV_HOST_ARCH AARCH64)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
  set(OV_HOST_ARCH ARM)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^riscv64$")
  set(OV_HOST_ARCH RISCV64)
endif()

macro(_ov_user_ext_detect_arch_by_processor_type)
  if(CMAKE_OSX_ARCHITECTURES AND APPLE)
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
      set(OV_ARCH AARCH64)
    elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
      set(OV_ARCH X86_64)
    elseif(CMAKE_OSX_ARCHITECTURES MATCHES ".*x86_64.*" AND CMAKE_OSX_ARCHITECTURES MATCHES ".*arm64.*")
      set(OV_ARCH UNIVERSAL2)
    else()
      message(FATAL_ERROR "Unsupported value: CMAKE_OSX_ARCHITECTURES = ${CMAKE_OSX_ARCHITECTURES}")
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(OV_ARCH X86_64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*|amd64.*|AMD64.*|wasm")
    set(OV_ARCH X86)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*|armv8)")
    set(OV_ARCH AARCH64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
    set(OV_ARCH ARM)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64$")
    set(OV_ARCH RISCV64)
  endif()
endmacro()

macro(_ov_user_ext_process_msvc_generator_platform)
  # if cmake -A <ARM|ARM64|x64|Win32> is passed
  if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    set(OV_ARCH AARCH64)
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM")
    set(OV_ARCH ARM)
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "x64")
    set(OV_ARCH X86_64)
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "Win32")
    set(OV_ARCH X86)
  else()
    _ov_user_ext_detect_arch_by_processor_type()
  endif()
endmacro()

if(MSVC64 OR MINGW64)
  _ov_user_ext_process_msvc_generator_platform()
elseif(MINGW OR (MSVC AND NOT CMAKE_CROSSCOMPILING))
  _ov_user_ext_process_msvc_generator_platform()
else()
  _ov_user_ext_detect_arch_by_processor_type()
endif()

set(HOST_${OV_HOST_ARCH} ON)
set(${OV_ARCH} ON)

unset(OV_ARCH)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(EMSCRIPTEN ON)
endif()

if(UNIX AND NOT (APPLE OR ANDROID OR EMSCRIPTEN OR CYGWIN))
    set(LINUX ON)
endif()
