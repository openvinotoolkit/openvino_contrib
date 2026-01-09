# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()

if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()

set(TARGET_NAME "user_ov_extensions")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(TBB COMPONENTS tbb)
find_package(OpenCV COMPONENTS core)

set(OP_REQ_TBB "complex_mul" "fft")

#
# Select specific operations
#

if(NOT CUSTOM_OPERATIONS)
  file(GLOB op_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
  file(GLOB op_dirs LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/*")
  list(REMOVE_ITEM op_dirs "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

  foreach(op IN LISTS op_src)
    get_filename_component(op_name ${op} NAME_WE)
    list(APPEND CUSTOM_OPERATIONS ${op_name})
  endforeach()

  foreach(op_dir IN LISTS op_dirs)
    if(IS_DIRECTORY "${op_dir}")
      get_filename_component(op_name "${op_dir}" NAME)
      list(APPEND CUSTOM_OPERATIONS ${op_name})
    endif()
  endforeach()

  # remove .cpp file with all extensions
  list(REMOVE_ITEM CUSTOM_OPERATIONS ov_extension)
endif()

list(APPEND SRC "${CMAKE_CURRENT_SOURCE_DIR}/ov_extension.cpp")

# filter out some operations, requiring specific dependencies

if(NOT OpenCV_FOUND)
  list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp")
  list(REMOVE_ITEM CUSTOM_OPERATIONS fft)
endif()

if(NOT TBB_FOUND)
  foreach(op IN LISTS OP_REQ_TBB)
    list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/${op}.cpp")
    list(REMOVE_ITEM CUSTOM_OPERATIONS ${op})
  endforeach()
endif()

message("  List of custom operations in ${TARGET_NAME} extension: ")
foreach(op IN LISTS CUSTOM_OPERATIONS)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${op}")
    file(GLOB op_src "${CMAKE_CURRENT_SOURCE_DIR}/${op}/*.cpp")
    list(APPEND SRC ${op_src})
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${op}.cpp")
    list(APPEND SRC "${CMAKE_CURRENT_SOURCE_DIR}/${op}.cpp")
  else()
    message(FATAL_ERROR "${TARGET_NAME} does not have operation with name '${op}'")
  endif()

  message("    - ${op}")
endforeach()

#
# Create library
#

add_library(${TARGET_NAME} SHARED ${SRC})

if(OpenCV_FOUND)
  target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES})
endif()

if(TBB_FOUND)
  target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_EXTENSION_API ${CUSTOM_OPERATIONS})

# TODO: remove
target_include_directories(${TARGET_NAME} PUBLIC ./include/)
