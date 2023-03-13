# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

include(CheckCXXCompilerFlag)

# to build only sentencepiece-static target
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
             PROPERTY EXCLUDE_FROM_ALL ON)

include(FetchContent)

FetchContent_Declare(
  sentencepiece
  URL      https://github.com/google/sentencepiece/archive/refs/tags/v0.1.97.tar.gz
  URL_HASH SHA256=41c3a07f315e3ac87605460c8bb8d739955bc8e7f478caec4017ef9b7d78669b
)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(cxx_flags "-Wno-undef")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4244: 'argument' : conversion from 'type1' to 'type2', possible loss of data
    # C4267: 'var' : conversion from 'size_t' to 'type', possible loss of data
    set(cxx_flags "/wd4244 /wd4267")
endif()

check_cxx_compiler_flag("-Wsuggest-override" SUGGEST_OVERRIDE_SUPPORTED)
if(SUGGEST_OVERRIDE_SUPPORTED)
    set(cxx_flags "${cxx_flags} -Wno-suggest-override")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxx_flags}")

FetchContent_MakeAvailable(sentencepiece)

# set include dirs for specific source files
target_include_directories(${TARGET_NAME} PRIVATE
  "${sentencepiece_SOURCE_DIR}/src/builtin_pb"
  "${sentencepiece_SOURCE_DIR}/src"
  "${sentencepiece_SOURCE_DIR}/third_party/protobuf-lite"
  "${sentencepiece_SOURCE_DIR}"
  "${sentencepiece_SOURCE_DIR}"
  "${sentencepiece_BINARY_DIR}")

# sentencepiece for some reason uses /MT on Windows, while default flag is /MD
if(CMAKE_CL_64)
    target_compile_options(sentencepiece-static PRIVATE "$<IF:$<CONFIG:Release>,/MD,/MDd>")
    target_compile_definitions(sentencepiece-static PRIVATE _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE sentencepiece-static)

# string_view is used from cxx17
string(REPLACE " " ";" cxx_flags "${cxx_flags}")
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17
                                                COMPILE_OPTIONS "${cxx_flags}")
