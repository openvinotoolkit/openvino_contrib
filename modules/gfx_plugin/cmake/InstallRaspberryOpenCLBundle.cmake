# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED CLVK_BUILD_DIR OR CLVK_BUILD_DIR STREQUAL "")
    message(FATAL_ERROR "CLVK_BUILD_DIR is required")
endif()
if(NOT DEFINED OUTPUT_DIR OR OUTPUT_DIR STREQUAL "")
    message(FATAL_ERROR "OUTPUT_DIR is required")
endif()

function(_gfx_find_required_file out_var file_name)
    file(GLOB_RECURSE _gfx_matches
        LIST_DIRECTORIES false
        "${CLVK_BUILD_DIR}/${file_name}")
    list(SORT _gfx_matches)
    if(NOT _gfx_matches)
        message(FATAL_ERROR "GFX: CLVK build did not produce ${file_name} under ${CLVK_BUILD_DIR}")
    endif()
    list(GET _gfx_matches 0 _gfx_match)
    set(${out_var} "${_gfx_match}" PARENT_SCOPE)
endfunction()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")

_gfx_find_required_file(_gfx_opencl_library "libOpenCL.so.0.1")
_gfx_find_required_file(_gfx_clspv_binary "clspv")

file(COPY "${_gfx_opencl_library}" DESTINATION "${OUTPUT_DIR}")
file(COPY "${_gfx_clspv_binary}" DESTINATION "${OUTPUT_DIR}")

file(GLOB_RECURSE _gfx_llvm_spirv_matches
    LIST_DIRECTORIES false
    "${CLVK_BUILD_DIR}/llvm-spirv")
if(_gfx_llvm_spirv_matches)
    list(SORT _gfx_llvm_spirv_matches)
    list(GET _gfx_llvm_spirv_matches 0 _gfx_llvm_spirv_binary)
    file(COPY "${_gfx_llvm_spirv_binary}" DESTINATION "${OUTPUT_DIR}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink libOpenCL.so.0.1 libOpenCL.so.1
    WORKING_DIRECTORY "${OUTPUT_DIR}"
    RESULT_VARIABLE _gfx_symlink_result)
if(NOT _gfx_symlink_result EQUAL 0)
    message(FATAL_ERROR "GFX: failed to create libOpenCL.so.1 symlink in ${OUTPUT_DIR}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink libOpenCL.so.1 libOpenCL.so
    WORKING_DIRECTORY "${OUTPUT_DIR}"
    RESULT_VARIABLE _gfx_symlink_result)
if(NOT _gfx_symlink_result EQUAL 0)
    message(FATAL_ERROR "GFX: failed to create libOpenCL.so symlink in ${OUTPUT_DIR}")
endif()

message(STATUS "GFX: staged Raspberry OpenCL bundle in ${OUTPUT_DIR}")
