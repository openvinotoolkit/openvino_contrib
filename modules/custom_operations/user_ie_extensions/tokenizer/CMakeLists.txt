# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

option(BUILD_FAST_TOKENIZERS OFF)

# to build only sentencepiece-static target
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL ON)

#
# Compile flags
#

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(cxx_flags "-Wno-undef")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4244: 'argument' : conversion from 'type1' to 'type2', possible loss of data
    # C4267: 'var' : conversion from 'size_t' to 'type', possible loss of data
    set(cxx_flags "/wd4244 /wd4267")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wsuggest-override" SUGGEST_OVERRIDE_SUPPORTED)
if(SUGGEST_OVERRIDE_SUPPORTED)
    set(cxx_flags "${cxx_flags} -Wno-suggest-override")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxx_flags}")

#
# Dependencies
#

include(FetchContent)

FetchContent_Declare(
  sentencepiece
  URL      https://github.com/google/sentencepiece/archive/refs/tags/v0.1.99.tar.gz
  URL_HASH SHA256=63617eaf56c7a3857597dcd8780461f57dd21381b56a27716ef7d7e02e14ced4
)
FetchContent_MakeAvailable(sentencepiece)

if(BUILD_FAST_TOKENIZERS)
  FetchContent_Declare(
    fast_tokenizer
    URL      https://github.com/PaddlePaddle/PaddleNLP/archive/refs/tags/v2.6.1.tar.gz
    URL_HASH SHA256=10e3489bc91e938c449a0448fa719e4536803ed6b1c1c95b3402430d6a8a221a
    PATCH_COMMAND git apply --ignore-whitespace "${CMAKE_CURRENT_LIST_DIR}/icu4c.patch"
  )

  FetchContent_GetProperties(fast_tokenizer)
  if(NOT fast_tokenizer_POPULATED)
    FetchContent_Populate(
      fast_tokenizer
    )
    set(WITH_PYTHON OFF CACHE BOOL "Disable Python API for fast_tokenizer")
    add_subdirectory(${fast_tokenizer_SOURCE_DIR}/fast_tokenizer ${CMAKE_CURRENT_BINARY_DIR}/fast_tokenizer)
  endif()

  # variables used later
  set(FAST_TOKENIZER_INCS
    "${fast_tokenizer_SOURCE_DIR}/fast_tokenizer"
    "${CMAKE_BINARY_DIR}/third_party/dart/src/extern_dart/include/"
    "${CMAKE_BINARY_DIR}/third_party/json/src/extern_json/single_include/"
    "${CMAKE_BINARY_DIR}/third_party/install/re2/include/")
  set(FAST_TOKENIZER_LIBS core_tokenizers)
else()
  if(WIN32 AND X86_64)
      FetchContent_Declare(
        fast_tokenizer
        URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-win-x64-1.0.2.zip
        URL_HASH SHA256=56470954014bdd3c8c8ad702d20f5f6aa5ab913bff92fd9c3c49ec6da31ff11d
      )
      FetchContent_Declare(
        re2
        URL      https://github.com/google/re2/archive/refs/tags/2022-04-01.tar.gz
        URL_HASH SHA256=1ae8ccfdb1066a731bba6ee0881baad5efd2cd661acd9569b689f2586e1a50e9
      )
      FetchContent_MakeAvailable(re2)
  elseif(LINUX AND X86_64)
      FetchContent_Declare(
        fast_tokenizer
        URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-x64-1.0.2.tgz
        URL_HASH SHA256=843a8299b55ef2e06ea50ba0d4ab4cb05b9e4cdb7cb8e29f3d55c494a1b7aecc
      )
  elseif(LINUX AND AARCH64)
      FetchContent_Declare(
        fast_tokenizer
        URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-aarch64-1.0.2.tgz
        URL_HASH SHA256=fc16c51b24a954ae3d659e1b233ce15349eafc1e4c72710b51a4f12fb2c03033
      )
  elseif(APPLE AND X86_64)
      FetchContent_Declare(
        fast_tokenizer
        URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-osx-x86_64-1.0.2.tgz
        # URL_HASH SHA256=4c8123ad941b3e4325ef72f328db545e34d5eec2de3e2545e1ab8ebeeb5146a9
      )
  elseif(APPLE AND AARCH64)
      FetchContent_Declare(
        fast_tokenizer
        URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-osx-arm64-1.0.2.tgz
        URL_HASH SHA256=ffb0f16ec96b2f5dbdb681d00d74e932e273ec1c2108196d13f2fd28abc4d266
      )
  else()
    message(FATAL_ERROR "Platform ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR} does not have prebuilt Fast Tokenizer"
                        "Please, use -DBUILD_FAST_TOKENIZERS=ON cmake option to enable build from soures")
  endif()
  FetchContent_MakeAvailable(fast_tokenizer)
  include("${fast_tokenizer_SOURCE_DIR}/FastTokenizer.cmake")

  if(WIN32 AND X86_64)
      # we use re2 library in regex_normalization operation, so have to add to this list
      # because prebuilt fast_tokenizers package does not provide this library
      list(APPEND FAST_TOKENIZER_LIBS re2)
  endif()
endif()

#
# Target include dirs, link libraries and other properties
#

# set include dirs for specific source files
target_include_directories(${TARGET_NAME} PRIVATE
  # sentensepiece
  "${sentencepiece_SOURCE_DIR}/src/builtin_pb"
  "${sentencepiece_SOURCE_DIR}/src"
  "${sentencepiece_SOURCE_DIR}/third_party/protobuf-lite"
  "${sentencepiece_SOURCE_DIR}"
  "${sentencepiece_BINARY_DIR}"
  # fast_tokenizer
  ${FAST_TOKENIZER_INCS})

if(CMAKE_CL_64)
    target_compile_definitions(sentencepiece-static PRIVATE _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE ${FAST_TOKENIZER_LIBS} sentencepiece-static)

# string_view is used from cxx17
string(REPLACE " " ";" cxx_flags "${cxx_flags}")
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17
                                                COMPILE_OPTIONS "${cxx_flags}")
