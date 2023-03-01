# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "user_ov_extensions")

find_package(OpenVINO REQUIRED)
find_package(TBB COMPONENTS tbb tbbmalloc)
find_package(OpenCV COMPONENTS core)

set(OP_REQ_TBB "complex_mul" "fft")

if(CUSTOM_OPERATIONS)
  file(GLOB SRC ov_extension.cpp)

  foreach(op IN LISTS CUSTOM_OPERATIONS)
    list(APPEND SRC ${CMAKE_CURRENT_SOURCE_DIR}/${op}.cpp)
  endforeach()
else()
  file(GLOB SRC *.cpp)

  foreach(op IN LISTS SRC)
    get_filename_component(op_name "${op}" NAME_WLE)
    list(APPEND CUSTOM_OPERATIONS ${op_name})
  endforeach()

  list(REMOVE_ITEM CUSTOM_OPERATIONS ov_extension)
endif()

if (NOT OpenCV_FOUND)
  list(REMOVE_ITEM SRC ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp)
  list(REMOVE_ITEM CUSTOM_OPERATIONS fft)
endif()

if (NOT TBB_FOUND)
  foreach(op IN LISTS OP_REQ_TBB)
    list(REMOVE_ITEM SRC ${CMAKE_CURRENT_SOURCE_DIR}/${op}.cpp)
    list(REMOVE_ITEM CUSTOM_OPERATIONS ${op})
  endforeach()
endif()

add_library(${TARGET_NAME} SHARED ${SRC})

if (OpenCV_FOUND AND "fft" IN_LIST CUSTOM_OPERATIONS)
  target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES})
endif()

if (TBB_FOUND)
  foreach(op IN LISTS OP_REQ_TBB)
    if(op IN_LIST CUSTOM_OPERATIONS)
      set(TBB_REQUIRED TRUE)
    endif()
  endforeach()

  if(TBB_REQUIRED)
    target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb TBB::tbbmalloc)
  endif()
endif()

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

target_compile_definitions(${TARGET_NAME} PRIVATE ${CUSTOM_OPERATIONS})
