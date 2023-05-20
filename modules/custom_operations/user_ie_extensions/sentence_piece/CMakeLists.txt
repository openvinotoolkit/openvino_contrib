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
  URL      https://github.com/google/sentencepiece/archive/87721596842ab099c603b23357d948906813e853.tar.gz
  URL_HASH SHA256=a7c105aca0131b4a899155a6c44ea9728e63514edaa8d71fa92e7a5de53b6ca0
)

FetchContent_Declare(
  fast_tokenizer
  URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-x64-1.0.2.tgz
  URL_HASH SHA256=843a8299b55ef2e06ea50ba0d4ab4cb05b9e4cdb7cb8e29f3d55c494a1b7aecc
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
FetchContent_MakeAvailable(fast_tokenizer)

include("${fast_tokenizer_SOURCE_DIR}/FastTokenizer.cmake")

# set include dirs for specific source files
target_include_directories(${TARGET_NAME} PRIVATE
  "${sentencepiece_SOURCE_DIR}/src/builtin_pb"
  "${sentencepiece_SOURCE_DIR}/src"
  "${sentencepiece_SOURCE_DIR}/third_party/protobuf-lite"
  "${sentencepiece_SOURCE_DIR}"
  "${sentencepiece_SOURCE_DIR}"
  "${sentencepiece_BINARY_DIR}"
  "${FAST_TOKENIZER_INCS}")


if(CMAKE_CL_64)
    target_compile_definitions(sentencepiece-static PRIVATE _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE sentencepiece-static ${FAST_TOKENIZER_LIBS})

# string_view is used from cxx17
string(REPLACE " " ";" cxx_flags "${cxx_flags}")
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17
                                                COMPILE_OPTIONS "${cxx_flags}")
