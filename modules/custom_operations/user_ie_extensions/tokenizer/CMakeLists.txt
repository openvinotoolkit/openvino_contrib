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
  URL      https://github.com/google/sentencepiece/archive/refs/tags/v0.1.99.tar.gz
  URL_HASH SHA256=63617eaf56c7a3857597dcd8780461f57dd21381b56a27716ef7d7e02e14ced4
)

FetchContent_Declare(
  fast_tokenizer
  URL      https://github.com/PaddlePaddle/PaddleNLP/archive/refs/tags/v2.6.1.tar.gz
  URL_HASH SHA256=10e3489bc91e938c449a0448fa719e4536803ed6b1c1c95b3402430d6a8a221a
)

FetchContent_GetProperties(fast_tokenizer)
if(NOT fast_tokenizer_POPULATED)
  FetchContent_Populate(
    fast_tokenizer
  )
  set(WITH_PYTHON OFF CACHE BOOL "Disable Python API for fast_tokenizer")
  add_subdirectory(${fast_tokenizer_SOURCE_DIR}/fast_tokenizer ${CMAKE_CURRENT_BINARY_DIR}/fast_tokenizer)
endif()

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
  # sentensepiece
  "${sentencepiece_SOURCE_DIR}/src/builtin_pb"
  "${sentencepiece_SOURCE_DIR}/src"
  "${sentencepiece_SOURCE_DIR}/third_party/protobuf-lite"
  "${sentencepiece_SOURCE_DIR}"
  "${sentencepiece_SOURCE_DIR}"
  "${sentencepiece_BINARY_DIR}"
  # fast_tokenizer
  "${fast_tokenizer_SOURCE_DIR}/fast_tokenizer"
  "${CMAKE_BINARY_DIR}/third_party/dart/src/extern_dart/include/"
  "${CMAKE_BINARY_DIR}/third_party/install/re2/include/")

target_include_directories(${TARGET_NAME} PUBLIC .)

if(CMAKE_CL_64)
    target_compile_definitions(sentencepiece-static PRIVATE _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE core_tokenizers sentencepiece-static)

# string_view is used from cxx17
string(REPLACE " " ";" cxx_flags "${cxx_flags}")
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17
                                                COMPILE_OPTIONS "${cxx_flags}")
